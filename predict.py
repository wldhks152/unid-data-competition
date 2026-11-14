"""추론 — 학습된 bundle로 test 예측 후 제출 파일 생성.
    python predict.py --data data --run lgb_v1 [--pp none|cummax|hmm] [--no-tune]

운영진이 '미래 관측을 제외한 입력'으로 재실행해 검증할 수 있으므로,
제출 전에 end-to-end 검사(테스트 시계열을 잘라도 앞부분 예측이 동일한지)를 자동으로 수행한다.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from src.data import KEY, T, load
from src.features import make_features
from src.model import predict_labels


def e2e_causality_check(bundle, te, pred_full, n_exp=5, seed=0, **kw):
    """test의 일부 실험을 임의 지점에서 잘라 다시 예측 → 남은 행의 예측이 전체 예측과 같아야 한다."""
    rng = np.random.default_rng(seed)
    exps = rng.choice(te[KEY].unique(), size=min(n_exp, te[KEY].nunique()), replace=False)
    for e in exps:
        g = te[te[KEY] == e].sort_values(T)
        cut = int(rng.integers(2, max(3, len(g))))
        Fp = make_features(g.iloc[:cut])
        p, _ = predict_labels(bundle, Fp, **kw)
        ref = pred_full.loc[Fp.sample_id].to_numpy()
        if not (p == ref).all():
            sys.exit(f"[누수!] 실험 {e}: 미래 관측을 빼면 예측이 달라집니다.")
    print(f"  E2E 누수 검사 OK — {len(exps)}개 실험을 잘라도 과거 예측 동일")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--run", default="lgb_v1")
    ap.add_argument("--out", default=None, help="기본: outputs/<run>/submission.csv")
    ap.add_argument("--no-tune", action="store_true", help="클래스 배율 w 미적용")
    ap.add_argument("--pp", choices=["none", "cummax", "hmm"], default="none",
                    help="후처리: cummax(예측 누적최대) / hmm(tune_pp.py 결과 사용)")
    ap.add_argument("--cummax", action="store_true", help="--pp cummax 와 같음 (하위 호환)")
    a = ap.parse_args()
    if a.cummax:
        a.pp = "cummax"
    run_dir = os.path.join("outputs", a.run)
    out = a.out or os.path.join(run_dir, f"submission_{a.pp}.csv")
    bundle = joblib.load(os.path.join(run_dir, "bundle.joblib"))
    hmm = None
    if a.pp == "hmm":
        pp_path = os.path.join(run_dir, "pp.json")
        if not os.path.exists(pp_path):
            sys.exit(f"[오류] {pp_path} 없음 → 먼저 python tune_pp.py --run {a.run}")
        with open(pp_path, encoding="utf-8") as f:
            hmm = json.load(f)
    kw = dict(use_w=not a.no_tune, cummax=a.pp == "cummax", hmm=hmm)

    d = load(a.data, ("test", "sub"))
    te = d["test"]
    F = make_features(te)
    pred, _ = predict_labels(bundle, F, **kw)
    pred_full = pd.Series(pred, index=F.sample_id.values)
    e2e_causality_check(bundle, te, pred_full, **kw)

    sub = d["sub"][["sample_id"]].copy()
    sub["label"] = pred_full.reindex(sub.sample_id).to_numpy()
    assert sub.label.notna().all() and len(sub) == len(d["sub"]), "제출 파일 행 불일치"
    sub["label"] = sub.label.astype(int)
    sub.to_csv(out, index=False)
    print(f"  저장: {out}  (pp={a.pp}, use_w={not a.no_tune})")
    print(f"  예측 분포: {sub.label.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
