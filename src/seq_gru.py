"""인과적 GRU 시퀀스 멤버 — 팀 레포(unid-data-competition)의 custom 멤버 규약을 따른다.

입력: 행마다 make_features 피처(센서 과거 피처) + 팔레트 이미지 통계(v2) + 회색조 이미지 통계(팀원) + 관측 간격 dt
모델: Linear → GRU(단방향) → Linear(6). 단방향 GRU는 t 시점 출력이 0..t 행만 보므로 미래 정보가 없다.
실험마다 따로 순전파하므로 다른 실험이 입력에 있든 없든 결과가 같다.
전처리 기준(중앙값·분위수 폭)은 fold 학습 데이터로만 계산해 저장한다.

학습: train_seq.py (OOF·fold 모델 → runs/seq). blend 번들에는 custom 멤버 {"module": "src.seq_gru"}로 들어간다.
"""
import numpy as np
import pandas as pd

KEY, T = "experiment_id", "relative_time_s"
CLIP = 5.0


# ---------------------------------------------------------------- 입력 행렬
def image_block(df, root=None, cache_dir=None, tag=None, timg=None):
    """sample_id 인덱스의 이미지 피처 (팔레트 v2 11개 + 회색조 6개). 행별 독립 계산."""
    from src.images import image_features
    from src.ensemble import team_image_features
    a = image_features(df, root, cache_dir=cache_dir, tag=tag)
    b = timg if timg is not None else team_image_features(df, root, cache_dir=cache_dir, tag=tag)
    b = b.copy()
    b.columns = [f"team_{c}" for c in b.columns]
    return pd.concat([a, b.reindex(a.index)], axis=1)


def assemble(F, img, feats):
    """F(make_features 결과)에 이미지 피처와 dt를 붙여 모델 입력 열 순서로 반환."""
    X = F.reindex(columns=feats).reset_index(drop=True)  # 없는 피처는 NaN → 학습 중앙값으로 대체됨
    I = img.reindex(F["sample_id"].values).reset_index(drop=True)
    t = F[T].to_numpy(float)
    same = np.r_[False, F[KEY].to_numpy()[1:] == F[KEY].to_numpy()[:-1]]
    dt = np.where(same, np.r_[0.0, np.diff(t)], 0.0)
    return pd.concat([X, I, pd.DataFrame({"dt_obs": dt})], axis=1)


def fit_scaler(X):
    A = X.to_numpy(np.float64)
    A = np.where(np.isfinite(A), A, np.nan)
    med = np.nanmedian(A, axis=0)
    lo, hi = np.nanpercentile(A, 5, axis=0), np.nanpercentile(A, 95, axis=0)
    scale = hi - lo
    sd = np.nanstd(A, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
    return {"cols": list(X.columns), "med": np.nan_to_num(med), "scale": scale}


def transform(X, sc):
    A = X[sc["cols"]].to_numpy(np.float64)
    A = np.where(np.isfinite(A), A, np.nan)
    Z = (A - sc["med"]) / sc["scale"]
    return np.clip(np.nan_to_num(Z, nan=0.0), -CLIP, CLIP).astype(np.float32)


def sequences(F, Z, y=None):
    """F는 (실험, 시간) 정렬 상태. 실험별 (index 배열, 입력, 라벨, dt) 목록."""
    F = F.reset_index(drop=True)
    dt = dt_scaled(F)
    out = []
    for _, idx in F.groupby(KEY, sort=False).indices.items():
        out.append((idx, Z[idx], None if y is None else y[idx], dt[idx]))
    return out


# ---------------------------------------------------------------- 모델
def build_net(n_in, hidden=128, layers=2, dropout=0.2, head="softmax"):
    """forward(x, dt) → 단계 1~6 로그확률 (B, L, 6).
    head="softmax": 시점별 독립 분류.
    head="hazard" : 시점별 비음수 위험률 g_j(t)≥0 (j=2..6)을 관측 간격 dt(초/5)에 곱해 시간 누적.
                    H_k(t) = Σ_{τ≤t} Σ_{j≥k} g_j(τ)·dt(τ),  P(단계≥k) = 1 − exp(−H_k)
                    → P(단계≥k)는 시간에 대해 단조 증가, k에 대해 단조 감소. 누적은 과거 방향만."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
            self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 6 if head == "softmax" else 5))
            if head == "hazard":
                nn.init.constant_(self.out[1].bias, -4.0)  # 초기 위험률을 작게

        def forward(self, x, dt=None):
            z = self.out(self.gru(self.inp(x))[0])
            if self.head == "softmax":
                return torch.log_softmax(z, -1)
            g = Fn.softplus(z) * dt.unsqueeze(-1)                       # (B, L, 5), j=2..6
            g = torch.flip(torch.cumsum(torch.flip(g, [-1]), -1), [-1])  # Σ_{j≥k}
            H = torch.cumsum(g, 1)                                     # 시간 누적 (과거 방향)
            surv = torch.exp(-H)                                       # P(단계<k), k=2..6
            one = torch.ones_like(surv[..., :1])
            lo = torch.cat([surv, one], -1)    # P(단계<k+1) for k=1..6 (k=6 → 1)
            hi = torch.cat([0 * one, surv], -1)  # P(단계<k)
            return torch.log((lo - hi).clamp_min(1e-7))

    return Net()


def dt_scaled(F):
    """관측 간격(초)/5. 실험 첫 행은 1."""
    t = F[T].to_numpy(float)
    k = F[KEY].to_numpy()
    same = np.r_[False, k[1:] == k[:-1]]
    return np.where(same, np.r_[0.0, np.diff(t)], 5.0).astype(np.float32) / 5.0


def predict_seqs(net, seqs, device="cpu"):
    """실험 하나씩 순전파 (패딩 없음 → 다른 실험과 섞일 일이 없음). seqs: (idx, z, y, dt)."""
    import torch
    net = net.to(device).eval()
    res = []
    with torch.no_grad():
        for idx, z, _, dt in seqs:
            lp = net(torch.from_numpy(z)[None].to(device), torch.from_numpy(dt)[None].to(device))[0]
            q = torch.exp(lp).double().cpu().numpy()
            res.append((idx, q / q.sum(1, keepdims=True)))
    return res


def load_net(state):
    import torch
    net = build_net(state["n_in"], **state["arch"])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in state["weights"].items()})
    return net


def proba_from_F(member, F, img):
    """F: make_features 결과(실험·시간 정렬, reset_index). fold 모델 확률 평균, 행 순서는 F와 같음."""
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    X = assemble(F, img, member["feats"])
    P = np.zeros((len(F), 6))
    for st in member["models"]:
        Z = transform(X, st["scaler"])
        net = load_net(st)
        for idx, p in predict_seqs(net, sequences(F, Z), device="cpu"):
            P[idx] += p
    P /= len(member["models"])
    return P / P.sum(1, keepdims=True)


def predict_proba(member, F, df, timg=None):
    """팀 레포 custom 멤버 진입점. 이미지 폴더는 test.py가 src.rn18.IMAGE_ROOT에 넣어 준다."""
    import src.rn18 as rn18
    root = df.attrs.get("image_root") or rn18.IMAGE_ROOT
    assert root, "seq_gru 멤버는 이미지 폴더가 필요합니다 (test.py --image_root)."
    F = F.reset_index(drop=True)
    img = image_block(df, root, timg=timg)
    return proba_from_F(member, F, img)
