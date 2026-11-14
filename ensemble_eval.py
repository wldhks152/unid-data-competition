"""비교 실험 — 우리 LightGBM(holdout)과 팀원 CatBoost v1을 제공 Validation에서 비교·앙상블.
두 모델 모두 Train만으로 학습된 것을 써야 공정하다.

준비 (팀원 브랜치를 옆 폴더로 꺼내기):
    git worktree add ../catboost_v1 origin/experiments/catboost_v1
실행:
    python ensemble_eval.py --data_dir <battery_data_light> --ours model_holdout.joblib \
                            --theirs_dir ../catboost_v1

출력: 우리 비율 α (0=팀원만, 1=우리만)별 Validation Macro F1 (raw / +w / cummax / +w+cummax, 5·10·15초)
"""
import argparse
import importlib.util
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import EVAL_VERSIONS, KEY, VER_W, Y, find_csv, subsample
from src.features import make_features
from src.model import cummax_pp

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_team_module(d):
    spec = importlib.util.spec_from_file_location("team_model", os.path.join(d, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ours", default="model_holdout.joblib", help="--holdout_valid 로 학습한 우리 모델")
    ap.add_argument("--theirs_dir", default="../catboost_v1", help="팀원 브랜치 폴더 (model.py, weights/)")
    ap.add_argument("--theirs_model", default=None, help="기본: <theirs_dir>/weights/model.joblib")
    ap.add_argument("--image_jobs", type=int, default=4)
    a = ap.parse_args()

    ours = joblib.load(a.ours)
    TM = load_team_module(a.theirs_dir)
    theirs = joblib.load(a.theirs_model or os.path.join(a.theirs_dir, "weights", "model.joblib"))
    classes = np.asarray(ours["classes"])
    t_classes = np.asarray(theirs["pipeline"].classes_)
    assert set(t_classes) == set(classes), (t_classes, classes)
    order = [list(t_classes).index(c) for c in classes]

    vpath = find_csv(a.data_dir, "valid")
    root = os.path.dirname(vpath)
    va_full = pd.read_csv(vpath, dtype={"sample_id": "string", "experiment_id": "string"})
    print(f"Validation: {len(va_full):,} rows, {va_full[KEY].nunique()} experiments")

    # 팀원 이미지 피처는 한 번만 계산해서 재사용 (행별 독립 계산이라 다운샘플해도 동일)
    paths = TM.resolve_image_paths(va_full, [root, os.path.dirname(root)])
    img_all = TM.extract_image_features(va_full, paths, n_jobs=a.image_jobs)

    res = {}
    for k, o in EVAL_VERSIONS:
        ver = f"k{k}o{o}"
        va = subsample(va_full, k, o) if k > 1 else va_full.assign(grp=va_full[KEY], ver=ver)
        F = make_features(va).reset_index(drop=True)
        P_o = np.mean([m.predict_proba(F[ours["feats"]]) for m in ours["models"]], axis=0)
        va_t = va.set_index("sample_id").loc[F.sample_id].reset_index()
        va_t["relative_time_s"] = F["relative_time_s"].values
        X_t = TM.build_features(va_t, img_all)[theirs["features"]]
        P_t = theirs["pipeline"].predict_proba(X_t)[:, order]
        y = F[Y].to_numpy()
        for al in ALPHAS:
            P = al * P_o + (1 - al) * P_t
            p_raw = classes[P.argmax(1)]
            p_w = classes[(P * ours["w"]).argmax(1)]
            for name, p in [("raw", p_raw), ("+w", p_w), ("cummax", cummax_pp(F, p_raw)),
                            ("+w+cummax", cummax_pp(F, p_w))]:
                res.setdefault((al, name), {})[ver] = f1_score(y, p, average="macro")
        print(f"  {5 * k}초 버전 완료")

    print("\n=== Validation Macro F1 (α = 우리 모델 비율, 0=팀원 CatBoost만, 1=우리만) ===")
    print(f"  {'α':>4s} {'후처리':<10s} {'5s':>7s} {'10s':>7s} {'15s':>7s} {'가중':>7s}")
    rows = []
    for (al, name), s in res.items():
        s["weighted"] = sum(VER_W[v] * s[v] for v in VER_W) / sum(VER_W.values())
        rows.append((al, name, s))
        print(f"  {al:>4.2f} {name:<10s} {s['k1o0']:7.4f} {s['k2o0']:7.4f} {s['k3o0']:7.4f} {s['weighted']:7.4f}")
    best = max(rows, key=lambda r: r[2]["weighted"])
    print(f"\n  최고: α={best[0]}, 후처리={best[1]}, 가중 {best[2]['weighted']:.4f}")


if __name__ == "__main__":
    main()
