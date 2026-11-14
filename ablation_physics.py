"""물리 피처 어블레이션 — 기존 피처(base)에 src/physics.py 그룹을 하나씩 더해 같은 5-fold로 비교한다.
    python ablation_physics.py --data_dir <battery_data_light 경로> [--fast] [--mode single|cumulative|both]

공정한 비교를 위해 전부 고정:
* 공통 fold(experiment_folds, 또는 --folds_json) · 같은 5/10/15초 증강
* class_weight 없음(--balanced로 켤 수 있음), 클래스 배율 튜닝 없음, 후처리 튜닝 없음
* 점수는 raw argmax. 참고용으로 '고정' cummax(튜닝 아님)도 같이 표시
* 2↔3 지표: 정답이 2 또는 3인 행에서 2/3 구분 정확도 (그 행들의 예측이 2·3 외인 경우는 오답)
산출물: <out>/ablation_summary.csv, ablation_folds.csv, ablation_importance.csv, folds.json,
        oof_<설정>.csv (VALIDATION.md 계약: make_oof/write_oof), --save_model이면 physics_lgb.joblib
fold · 점수 · OOF 형식은 src/evaluation.py 공통 모듈을 그대로 쓴다 (experiment_folds, score_fn, make_oof).
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import EVAL_VERSIONS, TRAIN_VERSIONS, VER_W, Y, load, subsample
from src.evaluation import CLASSES, PCOLS, experiment_folds, load_folds, make_oof, save_folds, score_fn, write_oof
from src.features import causality_check, feature_columns, make_features
from src.model import cummax_pp, make_model
from src.physics import GROUPS, group_of, parse_groups


def scores(dd, pred, classes):
    """dd: OOF(ver, y, ...). 간격별 Macro-F1, 공통 score_fn(9:1:1), 5초 클래스별 F1, 2↔3 정확도."""
    out = {}
    for v in VER_W:
        m = (dd.ver == v).to_numpy()
        out[f"f1_{v}"] = f1_score(dd.y[m], pred[m], average="macro", labels=CLASSES) if m.any() else np.nan
    out["f1_weighted"] = score_fn(dd, pred)  # 공통 점수 (9:1:1)
    m5 = (dd.ver == "k1o0").to_numpy()
    pc = f1_score(dd.y[m5], pred[m5], average=None, labels=classes)
    out.update({f"f1_c{c}": x for c, x in zip(classes, pc)})
    m23 = m5 & dd.y.isin([2, 3]).to_numpy()
    out["acc_2v3"] = float((pred[m23] == dd.y.to_numpy()[m23]).mean()) if m23.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "--data", dest="data_dir", required=True)
    ap.add_argument("--out", default="ablation_out")
    ap.add_argument("--seed", type=int, default=42, help="모델 seed (fold 분할은 공통 고정)")
    ap.add_argument("--groups", default="all", help=f"비교할 그룹 (쉼표). 가능: {','.join(GROUPS)}")
    ap.add_argument("--mode", default="both", choices=["single", "cumulative", "both"],
                    help="single: base+그룹 하나씩 · cumulative: 순서대로 누적 · both: 둘 다 (+전체)")
    ap.add_argument("--balanced", action="store_true", help="class_weight=balanced (기본 끔: 편향 없는 비교)")
    ap.add_argument("--fast", action="store_true", help="트리 수를 줄여 빠르게 (300개, lr 0.06)")
    ap.add_argument("--folds_json", default="", help="공통 folds.json (stack.py 산출물). 없으면 experiment_folds()")
    ap.add_argument("--save_model", action="store_true",
                    help="마지막 설정의 fold 모델을 blend.py용 custom 멤버(physics_lgb.joblib)로 저장")
    ap.add_argument("--save_oof", default="base,last", help="OOF 확률을 저장할 설정 (쉼표, base/last/all/설정이름)")
    ap.add_argument("--max_rows", type=int, default=0, help="(테스트용) 학습 행 수 상한")
    ap.add_argument("--cpu", action="store_true", help="CPU로 학습 (기본: GPU, LightGBM은 설치 빌드가 지원할 때)")
    a = ap.parse_args()
    if a.cpu:
        from src.model import DEVICE
        DEVICE["prefer"] = "cpu"
    os.makedirs(a.out, exist_ok=True)
    groups = parse_groups(a.groups)

    print("[1] 데이터 로드 (Train + Validation)")
    d = load(a.data_dir, ("train", "valid"))
    tv = pd.concat([d["train"], d["valid"]], ignore_index=True)

    print("\n[2] 피처 누수 검사 (기존 + 물리 피처 전체)")
    causality_check(tv, physics="all")

    print("\n[3] 피처 생성 (물리 피처 전체를 한 번만 계산)")
    t0 = time.time()
    F = pd.concat([make_features(subsample(tv, k, o), physics="all") for k, o in TRAIN_VERSIONS],
                  ignore_index=True)
    allf = [c for c in feature_columns(F) if F[c].nunique(dropna=True) > 1]
    base = [c for c in allf if group_of(c) is None]
    by_g = {g: [c for c in allf if group_of(c) == g] for g in GROUPS}
    print(f"  {len(F):,} rows, base {len(base)}개 + " + ", ".join(f"{g} {len(by_g[g])}" for g in GROUPS)
          + f" ({time.time() - t0:.0f}s)")

    configs = [("base", [])]
    if a.mode in ("single", "both"):
        configs += [(f"+{g}", [g]) for g in groups]
    if a.mode in ("cumulative", "both"):
        configs += [("+" + "+".join(groups[:i]), groups[:i]) for i in range(2, len(groups) + 1)]
    seen, uniq = set(), []
    for name, gs in configs:
        if tuple(gs) not in seen:
            seen.add(tuple(gs))
            uniq.append((name, gs))
    configs = uniq

    if a.folds_json:
        splits = load_folds(a.folds_json)
        print(f"  공통 fold 사용: {a.folds_json}")
    else:
        splits = experiment_folds(tv)
        print("  공통 fold 사용: src.evaluation.experiment_folds (seed 42, 실험 단위 5-fold)")
    save_folds(splits, os.path.join(a.out, "folds.json"))
    n_folds = len(splits)
    save_oof = set(a.save_oof.split(","))

    print(f"\n[4] 어블레이션 {len(configs)}개 설정 × {n_folds}-fold "
          f"(class_weight={'balanced' if a.balanced else '없음'}, 배율·후처리 튜닝 없음)")
    rows, frows, imps = [], [], []
    for ci, (name, gs) in enumerate(configs):
        feats = base + [c for g in gs for c in by_g[g]]
        t0 = time.time()
        oof = []
        models = []
        for fold, (tr_g, va_g) in enumerate(splits):
            trn = F[F.grp.isin(tr_g)]
            if a.max_rows and len(trn) > a.max_rows:
                trn = trn.sample(a.max_rows, random_state=a.seed)
            m = make_model(a.seed + fold, balanced=a.balanced)
            if a.fast and hasattr(m, "set_params") and "n_estimators" in m.get_params():
                m.set_params(n_estimators=300, learning_rate=0.06)
            m.fit(trn[feats], trn[Y])
            models.append(m)
            if hasattr(m, "feature_importances_") and gs:
                imps.append(pd.DataFrame({"config": name, "fold": fold, "feature": feats,
                                          "importance": m.feature_importances_}))
            for k, o in EVAL_VERSIONS:
                v = F[F.grp.isin(va_g) & (F.ver == f"k{k}o{o}")]
                P = m.predict_proba(v[feats])
                P6 = np.zeros((len(v), 6))
                for j, c in enumerate(m.classes_):  # 공통 형식: 클래스 순서 1..6
                    P6[:, int(c) - 1] = P[:, j]
                dd = make_oof(P6, v, f"k{k}o{o}", fold)
                oof.append(dd)
        oof = pd.concat(oof, ignore_index=True)
        oof["pred"] = CLASSES[oof[PCOLS].to_numpy().argmax(1)]
        tag = "base" if not gs else ("last" if ci == len(configs) - 1 else "")
        if "all" in save_oof or name in save_oof or tag in save_oof:
            fn = "oof_" + ("base" if not gs else "_".join(gs)) + ".csv"
            write_oof(oof.drop(columns=["pred"]), os.path.join(a.out, fn))
        if a.save_model and ci == len(configs) - 1 and gs:
            joblib.dump({"kind": "custom", "module": "src.physics_member", "models": models, "feats": feats,
                         "physics": gs, "classes": CLASSES}, os.path.join(a.out, "physics_lgb.joblib"), compress=3)
        classes = CLASSES
        for pp, pred in [("raw", oof.pred.to_numpy()), ("cummax", cummax_pp(oof, oof.pred.to_numpy()))]:
            s = scores(oof, pred, classes)
            rows.append({"config": name, "post": pp, "n_feats": len(feats), **s})
            for fold in range(n_folds):
                mk = (oof.fold == fold).to_numpy()
                sf = scores(oof[mk].reset_index(drop=True), pred[mk], classes)
                frows.append({"config": name, "post": pp, "fold": fold, **sf})
        r = rows[-2]
        print(f"  {name:<34s} feats={len(feats):4d}  raw 가중 {r['f1_weighted']:.4f}  "
              f"c2 {r.get('f1_c2', np.nan):.3f}  c3 {r.get('f1_c3', np.nan):.3f}  "
              f"2v3 {r['acc_2v3']:.3f}  | cummax {rows[-1]['f1_weighted']:.4f}  ({time.time() - t0:.0f}s)")

    S = pd.DataFrame(rows)
    Fo = pd.DataFrame(frows)
    # base 대비 fold별 차이 (같은 fold끼리 비교 → 잡음 판단용)
    b = Fo[Fo.config == "base"].set_index(["post", "fold"])
    for col in ["f1_weighted", "f1_c2", "f1_c3", "acc_2v3"]:
        if col not in Fo:
            continue
        diff = Fo.set_index(["post", "fold"])[col] - b[col].reindex(Fo.set_index(["post", "fold"]).index).values
        Fo[f"d_{col}"] = diff.values
    agg = Fo.groupby(["config", "post"]).agg(
        d_mean=("d_f1_weighted", "mean"), d_sd=("d_f1_weighted", "std"),
        folds_up=("d_f1_weighted", lambda x: int((x > 0).sum())),
        d_c2=("d_f1_c2", "mean"), d_c3=("d_f1_c3", "mean")).reset_index()
    S = S.merge(agg, on=["config", "post"], how="left")
    S.to_csv(os.path.join(a.out, "ablation_summary.csv"), index=False)
    Fo.to_csv(os.path.join(a.out, "ablation_folds.csv"), index=False)

    pd.set_option("display.width", 200)
    show = ["config", "n_feats", "f1_weighted", "f1_k1o0", "f1_k2o0", "f1_k3o0"] + \
        [c for c in S.columns if c.startswith("f1_c")] + ["acc_2v3", "d_mean", "d_sd", "folds_up"]
    for pp in ("raw", "cummax"):
        print(f"\n=== 요약 ({pp}{', 고정 후처리 · 참고용' if pp == 'cummax' else ', 후처리 없음'}) ===")
        print(S[S.post == pp][show].round(4).to_string(index=False))
    print(f"  d_mean/d_sd = base 대비 fold별 가중 F1 차이의 평균/표준편차, folds_up = 오른 fold 수 (/{n_folds})")

    if imps:
        I = pd.concat(imps)
        I = I[I.feature.map(group_of).notna()]
        top = I.groupby(["config", "feature"]).importance.mean().reset_index()
        top.to_csv(os.path.join(a.out, "ablation_importance.csv"), index=False)
        last = configs[-1][0]
        print(f"\n=== 물리 피처 중요도 Top 20 ({last}) ===")
        print(top[top.config == last].sort_values("importance", ascending=False).head(20)
              .round(1).to_string(index=False))
    print(f"\n저장 → {a.out}/ablation_summary.csv, ablation_folds.csv, ablation_importance.csv")


if __name__ == "__main__":
    main()
