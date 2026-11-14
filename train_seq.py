"""인과적 GRU 멤버 학습 — 팀 레포의 고정 fold(runs/stack/folds.json)로 OOF와 fold 모델을 만든다.

    python train_seq.py --data_dir <battery_data_light> --name seq2 --head hazard --seeds 3

산출물 (out_dir, 기본 runs/seq)
  oof_seq.csv  : 팀 레포 OOF 계약 형식 (blend.py --members ... seq=<out_dir>)
  seq.joblib   : {"kind": "custom", "module": "src.seq_gru", "models": [fold 5개], "feats": [...]}
  seq_cv.json  : 점수 요약
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import EVAL_VERSIONS, KEY, T, TRAIN_VERSIONS, find_csv, load, subsample
from src.evaluation import CLASSES, PCOLS, load_folds, make_oof, score_fn, tune_w, write_oof
from src.features import causality_check, feature_columns, make_features
from src.model import cummax_pp
from src import seq_gru as S

GATE_COLS = ["ev_t1_above45", "ev_v_below2.45", "ev_v_drop0.78", "ev_v_drop1.0", "tr_method"]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_fold(seqs, n_in, a, seed, device):
    set_seed(seed)
    net = S.build_net(n_in, hidden=a.hidden, layers=a.layers, dropout=a.dropout, head=a.head).to(device)
    y_all = np.concatenate([s[2] for s in seqs])
    freq = np.bincount(y_all - 1, minlength=6) / len(y_all)
    cw = torch.tensor((1 / np.maximum(freq, 1e-6)) ** a.cw_pow, dtype=torch.float32)
    cw = (cw / cw.mean()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * int(np.ceil(len(seqs) / a.batch))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    lossf = torch.nn.NLLLoss(weight=cw, reduction="none")
    rng = np.random.default_rng(seed)
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(len(seqs))
        tot = 0.0
        for b in range(0, len(order), a.batch):
            batch = [seqs[i] for i in order[b:b + a.batch]]
            L = max(len(s[1]) for s in batch)
            x = np.zeros((len(batch), L, n_in), np.float32)
            y = np.zeros((len(batch), L), np.int64)
            m = np.zeros((len(batch), L), np.float32)
            dts = np.ones((len(batch), L), np.float32)
            for j, (_, z, yy, dd) in enumerate(batch):
                n = len(z)
                zz = z + rng.normal(0, a.noise, z.shape).astype(np.float32) if a.noise > 0 else z
                x[j, :n], y[j, :n], m[j, :n], dts[j, :n] = zz, yy - 1, 1, dd
            x, y, m, dts = (torch.from_numpy(v).to(device) for v in (x, y, m, dts))
            logits = net(x, dts)  # 오른쪽 패딩 + 단방향 GRU → 실제 행의 출력은 패딩의 영향을 받지 않음
            loss = (lossf(logits.reshape(-1, 6), y.reshape(-1)) * m.reshape(-1)).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1:3d}  loss {tot / np.ceil(len(seqs) / a.batch):.4f}", flush=True)
    net = net.cpu().eval()
    return {"n_in": n_in, "arch": {"hidden": a.hidden, "layers": a.layers, "dropout": a.dropout, "head": a.head},
            "weights": {k: v.numpy() for k, v in net.state_dict().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "runs/seq"))
    ap.add_argument("--folds", default=os.path.join(REPO, "runs/stack/folds.json"))
    ap.add_argument("--cache_dir", default=os.path.join(REPO, "cache"))
    ap.add_argument("--name", default="seq")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cw_pow", type=float, default=0.5, help="클래스 가중치 (1/빈도)^pow")
    ap.add_argument("--noise", type=float, default=0.05, help="학습 입력 가우시안 노이즈 (표준화 단위)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=1, help="fold마다 시드 여러 개 학습해 확률 평균")
    ap.add_argument("--head", default="hazard", choices=["softmax", "hazard"])
    ap.add_argument("--max_folds", type=int, default=None, help="빠른 확인용")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[0] device={device}")

    print("[1] 데이터 · 이미지 피처 (캐시 사용)")
    d = load(a.data_dir, ("train", "valid"))
    tv = pd.concat([d["train"], d["valid"]], ignore_index=True)
    img = pd.concat([S.image_block(d[s], os.path.dirname(find_csv(a.data_dir, s)), a.cache_dir, s)
                     for s in ("train", "valid")])
    causality_check(tv)

    print("[2] 피처 (5/10/15초 버전)")
    t0 = time.time()
    fcache = os.path.join(a.out_dir, "F_trainval.pkl")
    if os.path.exists(fcache):
        F = pd.read_pickle(fcache)
    else:
        F = pd.concat([make_features(subsample(tv, k, o)) for k, o in TRAIN_VERSIONS], ignore_index=True)
        F.to_pickle(fcache)
    feats = [c for c in feature_columns(F) if F[c].nunique(dropna=True) > 1
             and c not in ("relative_time_s",)] + ["relative_time_s"]
    feats = list(dict.fromkeys(feats))
    X = S.assemble(F, img, feats)
    y = F["label"].to_numpy().astype(int)
    print(f"  {len(F):,} rows × {X.shape[1]} inputs ({time.time() - t0:.0f}s)")

    splits = load_folds(a.folds)
    if a.max_folds:
        splits = splits[:a.max_folds]
    models, oof = [], []
    for fold, (tr_g, va_g) in enumerate(splits):
        t0 = time.time()
        trm = F.grp.isin(tr_g).to_numpy()
        sc = S.fit_scaler(X[trm])
        Z = S.transform(X, sc)
        tr_seqs = [s for s in S.sequences(F, Z, y) if trm[s[0][0]]]
        print(f"  fold{fold}: train seqs {len(tr_seqs)}", flush=True)
        nets = []
        for si in range(a.seeds):
            st = train_fold(tr_seqs, Z.shape[1], a, a.seed + fold + 1000 * si, device)
            st["scaler"] = sc
            models.append(st)
            nets.append(S.load_net(st))
        msg = []
        for k, o in EVAL_VERSIONS:
            ver = f"k{k}o{o}"
            vm = (F.grp.isin(va_g) & (F.ver == ver)).to_numpy()
            Fv = F[vm].reset_index(drop=True)
            P = np.zeros((len(Fv), 6))
            for net in nets:
                for idx, p in S.predict_seqs(net, S.sequences(Fv, Z[vm]), device=device):
                    P[idx] += p / len(nets)
            dd = make_oof(P, Fv, ver, fold, extra_cols=GATE_COLS)
            oof.append(dd)
            from sklearn.metrics import f1_score
            msg.append(f"{5 * k}s {f1_score(dd.y, CLASSES[P.argmax(1)], average='macro'):.3f}")
        print(f"  fold{fold}  " + "  ".join(msg) + f"  ({time.time() - t0:.0f}s)", flush=True)

    oof = pd.concat(oof, ignore_index=True)
    P = oof[PCOLS].to_numpy()
    raw = CLASSES[P.argmax(1)]
    res = {"raw": score_fn(oof, raw, detail=True, require_all=not a.max_folds),
           "cummax": score_fn(oof, cummax_pp(oof, raw), detail=True, require_all=not a.max_folds)}
    if not a.max_folds:
        w, _ = tune_w(P, oof, post=lambda p: cummax_pp(oof, p))
        res["w_cummax"] = score_fn(oof, cummax_pp(oof, CLASSES[(P * w).argmax(1)]), detail=True)
        res["w"] = w.tolist()
        write_oof(oof, os.path.join(a.out_dir, f"oof_{a.name}.csv"))
        joblib.dump({"kind": "custom", "module": "src.seq_gru", "models": models, "feats": feats},
                    os.path.join(a.out_dir, f"{a.name}.joblib"), compress=3)
    conv = lambda v: {str(k): (round(float(x), 4) if np.isscalar(x) else conv(x)) for k, x in v.items()} if isinstance(v, dict) else v
    with open(os.path.join(a.out_dir, f"{a.name}_cv.json"), "w") as f:
        json.dump({"args": vars(a), **{k: conv(v) for k, v in res.items()}}, f, indent=1, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:9s} 가중 {v['weighted']:.4f}  클래스별(5s) {np.round(list(v['class_5s'].values()), 3)}")


if __name__ == "__main__":
    main()
