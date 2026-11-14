"""처음부터 학습하는 시퀀스 모델 — 실험 단위 5-fold OOF (기존 LightGBM/CatBoost OOF와 섞을 수 있는 형식).
    python train_seq.py --data_dir <battery_data_light> --model gru --out_dir runs/seq      # → seq_gru
    python train_seq.py --synthetic 30 --model rocket --out_dir runs/syn               # 합성 데이터로 동작 확인

* 사전학습 가중치·외부 데이터 없음. 모든 모델은 같은 실험의 현재·과거 행만 사용 (실행 시 자동 누수 검사)
* fold: VALIDATION.md 계약 — --folds_json(runs/stack/folds.json)이 있으면 load_folds, 없으면 experiment_folds(tv)
* 학습: Train 실험을 5/10/15초로 솎아낸 6개 버전(TRAIN_VERSIONS)을 함께 학습 + 입력에 Δt 채널
* 산출물 (out_dir 안, name 기본 seq_<model>):
    oof_<name>.csv  계약 형식 (sample_id, ver, grp, fold, y, experiment_id, relative_time_s, p1..p6) — write_oof로 검사
    <name>.joblib   custom 멤버 {"kind": "custom", "module": "seq.member", "models": fold×seed 모델}
    <name>_cv.json  score_fn 점수(5/10/15초 9:1:1 가중, fold별, 클래스별)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from seq import core
import joblib

from seq.prep import build_sequences, versions
from src.data import EVAL_VERSIONS, KEY, T, TRAIN_VERSIONS, load
from src.evaluation import CLASSES, experiment_folds, load_folds, make_oof, score_fn, write_oof

DEFAULTS = {
    "gru": {"epochs": 40, "lr": 2e-3, "batch": 8, "wd": 1e-2, "noise": 0.05, "net_kw": {"hid": 128, "layers": 2, "drop": 0.2}},
    "tcn": {"epochs": 40, "lr": 2e-3, "batch": 8, "wd": 1e-2, "noise": 0.05, "net_kw": {"ch": 96, "drop": 0.15}},
    "rocket": {"epochs": 30, "lr": 1e-3, "wd": 1e-4, "max_rows": 300000, "window": 24, "hidden": 0},
}


def cummax_by_exp(oof, pred):
    s = pd.Series(pred, index=oof.index)
    o = oof.sort_values([KEY, T]).index
    return s.loc[o].groupby(oof.loc[o, KEY].values).cummax().loc[oof.index].to_numpy()


def report(oof):
    P = oof[[f"p{c}" for c in CLASSES]].to_numpy()
    raw = CLASSES[P.argmax(1)]
    res = {}
    for name, pred in [("raw", raw), ("cummax", cummax_by_exp(oof, raw))]:
        s = score_fn(oof, pred, detail=True)
        res[name] = {k: (round(float(v), 4) if not isinstance(v, dict) else {kk: round(float(vv), 4) for kk, vv in v.items()})
                     for k, v in s.items()}
        print(f"  {name:<8s} 5s {s['k1o0']:.4f}  10s {s['k2o0']:.4f}  15s {s['k3o0']:.4f}  가중 {s['weighted']:.4f}")
        print(f"           클래스별 F1 (5s): {res[name]['class_5s']}")
        print(f"           fold별 F1 (5s):  {res[name]['fold_5s']}")
    return res


def causality_check(models, seqs, device, n=4, seed=0, atol=2e-4):
    """일부 실험을 임의 지점에서 잘라 다시 예측 → 남은 행 확률이 전체 예측과 같아야 한다."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    full = core.predict_ensemble(models, [seqs[i] for i in pick], device)
    worst = 0.0
    for P, i in zip(full, pick):
        s = seqs[i]
        cut = int(rng.integers(1, len(s["X"]) + 1))
        part = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        Pp = core.predict_ensemble(models, [part], device)[0]
        worst = max(worst, float(np.abs(Pp - P[:cut]).max()))
    if worst > atol:
        raise SystemExit(f"[누수!] 미래 행을 빼면 확률이 달라집니다 (최대 차이 {worst:.2e})")
    print(f"  누수 검사 OK — {len(pick)}개 실험을 잘라도 남은 행 확률 동일 (최대 차이 {worst:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--synthetic", type=int, default=0, help="합성 실험 N개로 동작 확인 (data_dir 무시)")
    ap.add_argument("--model", choices=list(DEFAULTS), default="gru")
    ap.add_argument("--out_dir", default="runs/seq")
    ap.add_argument("--name", default=None, help="멤버 이름 (기본 seq_<model>) → oof_<name>.csv, <name>.joblib")
    ap.add_argument("--folds_json", default=None, help="stack.py가 저장한 runs/stack/folds.json (없으면 experiment_folds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_seeds", type=int, default=1, help="fold마다 seed 몇 개를 평균할지")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--cw23", type=float, default=None, help="2·3단계 클래스 가중 배율 (GRU/TCN)")
    ap.add_argument("--aux23", type=float, default=None, help="2↔3 경계 보조 손실 비중 (GRU/TCN)")
    ap.add_argument("--extra_ch", action="store_true", help="핵심 센서 15초·60초 기울기 채널 추가 (seq/prep.py _extra_channels)")
    ap.add_argument("--ordinal", action="store_true", help="출력층을 순서형(누적 로짓)으로 (GRU/TCN)")
    ap.add_argument("--hidden", type=int, default=None, help="rocket: 0이면 선형, >0이면 은닉층 1개")
    ap.add_argument("--only_folds", type=int, nargs="*", default=None, help="일부 fold만 (빠른 확인용)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--holdout_valid", action="store_true",
                    help="--folds_json(Train 실험만)과 함께: Validation 실험을 마지막 fold로 두고 Train 전체로 학습해 예측 (stack.py --holdout_valid와 같은 형식)")
    a = ap.parse_args()

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if a.device == "auto" else a.device))
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("device: cpu")
    cfg = dict(DEFAULTS[a.model], model=a.model)
    for k in ("epochs", "lr", "batch", "hidden", "cw23", "aux23"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.ordinal:
        cfg["ordinal"] = True
    if a.extra_ch:
        cfg["extra_ch"] = True
    name = a.name or f"seq_{a.model}"
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, name)

    print("[1] 데이터")
    if a.synthetic:
        from seq.synth import make_synthetic
        tv = make_synthetic(a.synthetic, seed=a.seed)
    else:
        d = load(a.data_dir, ("train", "valid"))
        tv = pd.concat([d["train"], d["valid"]], ignore_index=True)
    tv["sample_id"] = tv["sample_id"].astype(str)
    print(f"  {len(tv):,} rows, {tv[KEY].nunique()} experiments")

    splits = load_folds(a.folds_json) if a.folds_json else experiment_folds(tv)
    print(f"  fold: {a.folds_json or 'experiment_folds(tv)'}")
    fold_of = {g: f for f, (_, va) in enumerate(splits) for g in va}
    if a.holdout_valid:
        if a.synthetic or not a.folds_json:
            raise SystemExit("--holdout_valid는 실제 데이터와 --folds_json(Train 실험만의 fold)이 필요합니다")
        vex = set(d["valid"][KEY])
        assert not vex & set(fold_of), "folds_json에 Validation 실험이 들어 있습니다 (Train만으로 만든 fold여야 함)"
        fold_of.update({g: len(splits) for g in vex})
        print(f"  Validation 실험 {len(vex)}개 → fold {len(splits)} (Train 전체로 학습해 예측)")
    miss = set(tv[KEY]) - set(fold_of)
    if miss:
        raise SystemExit(f"fold에 없는 실험 {len(miss)}개: {list(miss)[:5]}")
    n_folds = len(splits) + (1 if a.holdout_valid else 0)

    print("[2] 시퀀스 만들기 (5/10/15초 버전)")
    t0 = time.time()
    vers = sorted(set(TRAIN_VERSIONS) | set(EVAL_VERSIONS))
    seqs = build_sequences(versions(tv, vers), extra=cfg.get("extra_ch", False))
    print(f"  {len(seqs)} 시퀀스, 채널 {seqs[0]['X'].shape[1]}개, 최대 길이 {max(len(s['X']) for s in seqs)} ({time.time() - t0:.0f}s)")
    train_v = {f"k{k}o{o}" for k, o in TRAIN_VERSIONS}
    eval_v = {f"k{k}o{o}" for k, o in EVAL_VERSIONS}

    print(f"[3] {n_folds}-fold 학습 ({a.model}, cfg={cfg})")
    oof, all_models = [], []
    for f in range(n_folds):
        if a.only_folds is not None and f not in a.only_folds:
            continue
        hold = len(splits) if a.holdout_valid else -1  # Validation은 어떤 fold의 학습에도 넣지 않는다
        tr = [s for s in seqs if fold_of[s["grp"]] not in (f, hold) and s["ver"] in train_v]
        va = [s for s in seqs if fold_of[s["grp"]] == f and s["ver"] in eval_v]
        t0 = time.time()
        models = [core.fit(cfg, tr, device, a.seed + 100 * f + i) for i in range(a.n_seeds)]
        all_models += models
        P = core.predict_ensemble(models, va, device)
        for s, p in zip(va, P):
            fr = pd.DataFrame({"sample_id": s["sample_id"], "grp": s["grp"], "label": s["y"], KEY: s["key"], T: s["t"]})
            oof.append(make_oof(p, fr, s["ver"], f))
        m5 = [(s, p) for s, p in zip(va, P) if s["ver"] == "k1o0"]
        yy = np.concatenate([s["y"] for s, _ in m5])
        pp = CLASSES[np.concatenate([p for _, p in m5]).argmax(1)]
        print(f"  fold{f}  학습 {len(tr)} / 검증 {len(va)} 시퀀스  5s raw F1={f1_score(yy, pp, average='macro'):.4f}  ({time.time() - t0:.0f}s)")
        if f == (a.only_folds or [0])[0]:
            causality_check(models, va, device)

    oof = pd.concat(oof, ignore_index=True)
    print("\n[4] OOF Macro F1 (score_fn: 5/10/15초 9:1:1 가중)")
    res = report(oof)
    write_oof(oof, os.path.join(a.out_dir, f"oof_{name}.csv"))
    member = {"kind": "custom", "module": "seq.member", "name": name, "cfg": cfg, "models": all_models}
    joblib.dump(member, f"{out}.joblib", compress=3)
    with open(f"{out}_cv.json", "w", encoding="utf-8") as fp:
        json.dump({"args": vars(a), "cfg": cfg, "score": res, "n_rows": len(oof)}, fp, ensure_ascii=False, indent=2)
    print(f"\n저장 → {a.out_dir}/oof_{name}.csv, {out}.joblib, {out}_cv.json")


if __name__ == "__main__":
    main()
