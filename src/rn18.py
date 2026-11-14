"""팀원 ResNet18 + CatBoost v1을 스태킹 멤버(rn18)로 — 같은 fold로 학습해 OOF를 만든다.

* 이미지 특징: ImageNet 사전학습 ResNet18 (torchvision ResNet18_Weights.IMAGENET1K_V1, 동결, avgpool 512차원).
  가중치 출처 https://download.pytorch.org/models/resnet18-f37072fd.pth — 대회 규정 4(대회 전 공개된 범용
  사전학습 모델 허용, README에 출처 기재)에 따라 사용. 구현·전처리는 src/rn18_encoder.py (팀원 코드, BSD-3 고지는
  third_party/LICENSE.torchvision), 가중치 조각은 weights/encoder/.
* 표 특징: 팀원 CatBoost v1의 33개 (src/team_catboost.build_features) + 512개 = 545개.
* 이미지 특징은 행(이미지)마다 독립 계산 → 미래 정보 없음, 캐시해도 안전.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import team_catboost as TM

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "weights" / "encoder" / "manifest.json"
EMB_COLS = [f"resnet18_{i:03d}" for i in range(512)]
# 팀원 v2 (experiments/resnet18_catboost_v2): 현재 행의 센서 13개를 그대로 덧붙인다
ADDED_SENSORS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o",
                 "meas_no", "meas_no2", "meas_so2", "thermal_t", "atmp_t", "indu_t"]
CLASSES = np.arange(1, 7)
_ENC = {}
_MEMO = {}          # 프로세스 안 경로별 임베딩 메모 (같은 이미지 파일을 다시 읽지 않음)
IMAGE_ROOT = None   # test.py가 --image_root로 채운다


def _encoder(device="auto"):
    if device not in _ENC:
        from .rn18_encoder import FrozenResNet18
        _ENC[device] = FrozenResNet18(MANIFEST, device)
    return _ENC[device]


BATCH = 256


def _extract(enc, paths, label, batch=BATCH, workers=None):
    """배치 추출. 전처리(PNG 열기·리사이즈)는 스레드로 병렬, 마지막 배치도 0으로 채워 항상 같은 배치 크기로 돌린다
    → 행 순서·실험 분할이 달라도 이미지마다 같은 임베딩 (cuDNN 알고리즘이 배치 크기에 따라 바뀌지 않음)."""
    from concurrent.futures import ThreadPoolExecutor
    from .rn18_encoder import _torch, preprocess_image
    torch = _torch()
    out = np.empty((len(paths), 512), dtype=np.float32)
    workers = workers or min(16, os.cpu_count() or 4)
    with ThreadPoolExecutor(workers) as ex, torch.inference_mode():
        for s in range(0, len(paths), batch):
            chunk = paths[s:s + batch]
            x = np.zeros((batch, 3, 224, 224), dtype=np.float32)
            x[:len(chunk)] = np.stack(list(ex.map(lambda p: preprocess_image(Path(p)), chunk)))
            f = enc.model(torch.from_numpy(x).to(enc.device)).float().cpu().numpy()[:len(chunk)]
            if not np.isfinite(f).all():
                raise ValueError("ResNet18 특징에 NaN/inf")
            out[s:s + len(chunk)] = f
            done = s + len(chunk)
            if done % (batch * 20) < batch or done == len(paths):
                print(f"[{label}] {done:,}/{len(paths):,} images (batch={batch})", flush=True)
    return out


def embeddings(df, image_root, cache_dir=None, tag=None, device="auto"):
    """sample_id 인덱스의 (n, 512) DataFrame. cache_dir·tag가 있으면 npz로 캐시."""
    cache = os.path.join(cache_dir, f"rn18_{tag}.npz") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        E = pd.DataFrame(z["E"], index=pd.Index(z["ids"], name="sample_id"), columns=EMB_COLS)
        if set(df.sample_id) <= set(E.index):
            return E.loc[df.sample_id.values]
    root = Path(image_root)
    paths = [str(Path(p).resolve()) for p in TM.resolve_image_paths(df, [root, root.parent])]
    todo = sorted({p for p in paths if p not in _MEMO})
    if todo:
        for p, e in zip(todo, _extract(_encoder(device), todo, tag or "images")):
            _MEMO[p] = e
    E = np.stack([_MEMO[p] for p in paths]).astype(np.float32) if paths else np.zeros((0, 512), np.float32)
    out = pd.DataFrame(E, index=pd.Index(df.sample_id.values, name="sample_id"), columns=EMB_COLS)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, E=E, ids=np.asarray(df.sample_id.values, dtype=object))
    return out


DYN_WINDOWS = (15.0, 30.0, 60.0, 120.0)  # 초 단위 과거 창 (행 수가 아님 → 5/10/15초 간격에서 같은 의미)
DYN_COLS = (["dyn_first"] + [f"dyn_w{int(w)}" for w in DYN_WINDOWS] + ["dyn_rate15", "dyn_rate30", "dyn_base", "dyn_novel",
            "dyn_first_cmax", "dyn_w30_cmax", "dyn_novel_cmax", "dyn_t_since_w30_peak", "dyn_norm_d0"])


def emb_dynamics(frame, E, delta=False):
    """이미지 임베딩의 '시간 변화' 피처 — 같은 실험의 현재·과거 프레임만 쓴다 (미래 정보 없음).
    2→3 전환은 센서 임계값 규칙이 없어서(일치율 최고 33%), 화면 자체가 바뀌는 순간(연기·벤팅 등)을 잡으려는 것.
      dyn_first : 첫 프레임과의 코사인 거리          dyn_w{15,30,60,120}: w초 전 프레임과의 코사인 거리
      dyn_rate  : 그 거리 / 실제 시간차              dyn_base : 처음 60초 평균 프레임과의 거리
      dyn_novel : 15초 이상 지난 과거 프레임 중 가장 비슷한 것과의 거리 (처음 보는 화면일수록 큼)
      *_cmax    : 지금까지의 최댓값                  dyn_t_since_w30_peak: 30초 변화가 가장 컸던 때부터 경과초
    행별 내적은 einsum으로 행마다 따로 계산 → 뒤 행을 지우거나 행 순서를 섞어도 같은 값.
    delta=True면 (현재 임베딩 − 첫 프레임 임베딩) 전체도 함께 반환."""
    from .data import KEY, T
    E = np.asarray(E, np.float64)
    n = len(frame)
    U = E / np.maximum(np.sqrt(np.einsum("ij,ij->i", E, E)), 1e-12)[:, None]
    nrm = np.sqrt(np.einsum("ij,ij->i", E, E))
    out = {c: np.full(n, np.nan) for c in DYN_COLS}
    D = np.zeros_like(E) if delta else None
    t_all = pd.to_numeric(frame[T], errors="coerce").to_numpy(float)
    dist = lambda A, B: 1.0 - np.einsum("ij,ij->i", A, B)
    for idx in frame.groupby(KEY, sort=False).indices.values():
        idx = np.asarray(idx)
        o = idx[np.argsort(t_all[idx], kind="stable")]
        t, X, m = t_all[o], U[o], len(o)
        c = {"dyn_first": dist(X, np.repeat(X[:1], m, axis=0)), "dyn_norm_d0": nrm[o] - nrm[o][0]}
        for w in DYN_WINDOWS:
            j = np.searchsorted(t, t - w, side="left")
            dt = t - t[j]
            d = np.where(dt > 0, dist(X, X[j]), np.nan)
            c[f"dyn_w{int(w)}"] = d
            if w in (15.0, 30.0):
                c[f"dyn_rate{int(w)}"] = d / np.where(dt > 0, dt, np.nan)
        inb = t <= t[0] + 60.0
        mu = np.cumsum(np.where(inb[:, None], X, 0.0), axis=0) / np.cumsum(inb)[:, None]
        mu = mu / np.maximum(np.sqrt(np.einsum("ij,ij->i", mu, mu)), 1e-12)[:, None]
        c["dyn_base"] = dist(X, mu)
        nov = np.full(m, np.nan)
        jmax = np.searchsorted(t, t - 15.0, side="right")  # 15초 이상 지난 프레임 수
        for s in range(0, m, 512):
            e = min(m, s + 512)
            S = np.einsum("id,jd->ij", X[s:e], X[:jmax[e - 1]]) if jmax[e - 1] > 0 else np.zeros((e - s, 0))
            S = np.where(np.arange(S.shape[1])[None, :] < jmax[s:e, None], S, -np.inf)
            if S.shape[1]:
                best = S.max(1)
                nov[s:e] = np.where(jmax[s:e] > 0, 1.0 - best, np.nan)
        c["dyn_novel"] = nov
        with np.errstate(invalid="ignore"):
            for a in ("dyn_first", "dyn_w30", "dyn_novel"):
                c[a + "_cmax"] = np.fmax.accumulate(c[a])
            d30 = c["dyn_w30"]
            is_new = d30 >= c["dyn_w30_cmax"]
        last = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
        c["dyn_t_since_w30_peak"] = np.where(np.isnan(c["dyn_w30_cmax"]), np.nan, t - t[last])
        for k, v in c.items():
            out[k][o] = v
        if delta:
            D[o] = E[o] - E[o][0]
    res = pd.DataFrame(out, columns=DYN_COLS).astype(np.float32)
    if delta:
        res = pd.concat([res, pd.DataFrame(D.astype(np.float32), columns=[f"dd_{i:03d}" for i in range(E.shape[1])])], axis=1)
    return res


def build_X(frame, timg, emb, sensors=False, dyn=0):
    """sensors=False: v1 (33 + 512), True: v2 (33 + 512 + 현재 센서 13).
    dyn=1: + 이미지 임베딩 시간 변화 피처(emb_dynamics), dyn=2: + 첫 프레임 대비 임베딩 차이 전체."""
    base = TM.build_features(frame, timg)
    E = emb.reindex(frame["sample_id"].values)
    assert E.notna().all(axis=1).all(), "이미지 임베딩이 없는 샘플이 있습니다."
    parts = [base.reset_index(drop=True), pd.DataFrame(E.to_numpy(np.float32), columns=list(emb.columns))]
    if sensors:
        parts.append(TM.numeric(frame, ADDED_SENSORS).reset_index(drop=True))
    if dyn:
        parts.append(emb_dynamics(frame.reset_index(drop=True), E.to_numpy(np.float32), delta=dyn >= 2))
    return pd.concat(parts, axis=1)


def fit(raw, timg, emb, thread_count=None, gpu=True, sensors=False, seed=None, dyn=0):
    """팀원 CatBoost v1 설정 그대로, gpu=True면 CatBoost를 GPU로 학습한다 (545개 특징이라 CPU는 느림)."""
    X = build_X(raw, timg, emb, sensors, dyn)
    pipe = TM.build_model(thread_count=thread_count or os.cpu_count() or 4)
    if gpu and "classifier" in pipe.named_steps and type(pipe.named_steps["classifier"]).__name__ == "CatBoostClassifier":
        pipe.set_params(classifier__task_type="GPU", classifier__devices="0")
    if seed is not None and "classifier" in pipe.named_steps and type(pipe.named_steps["classifier"]).__name__ == "CatBoostClassifier":
        pipe.set_params(classifier__random_seed=int(seed))  # 시드 변형(_s1, _s2 …)용. None이면 팀원 설정(42) 그대로
    pipe.fit(X, raw["label"].astype(int).to_numpy())
    return {"features": list(X.columns), "pipeline": pipe, "sensors": sensors, "dyn": dyn}


def proba(model, raw, timg, emb, sample_ids):
    frame = raw.set_index("sample_id").loc[list(sample_ids)].reset_index()
    X = build_X(frame, timg, emb, model.get("sensors", False), model.get("dyn", 0))[model["features"]]
    P = model["pipeline"].predict_proba(X)
    tc = [int(c) for c in model["pipeline"].classes_]
    return P[:, [tc.index(int(c)) for c in CLASSES]]


def predict_proba(member, F, df, timg=None):
    """blend.py custom 멤버 추론. test.py가 IMAGE_ROOT(또는 df.attrs["image_root"])에 이미지 폴더를 넣어 준다."""
    root = df.attrs.get("image_root") or IMAGE_ROOT
    assert root, "rn18 멤버는 이미지 폴더가 필요합니다 (test.py --image_root)."
    if timg is None:
        from .ensemble import team_image_features
        timg = team_image_features(df, root)
    if member.get("embedder") == "dino":  # DINOv2 이미지 특징 (src/dino.py)
        from . import dino
        emb = dino.embeddings(df, root, device=member.get("device", "auto"))
    elif member.get("embedder") == "r50":  # ResNet50 이미지 특징 (src/r50.py)
        from . import r50
        emb = r50.embeddings(df, root, device=member.get("device", "auto"))
    else:
        emb = embeddings(df, root, device=member.get("device", "auto"))
    return np.mean([proba(m, df, timg, emb, F.sample_id.values) for m in member["models"]], axis=0)
