"""시드 변형 멤버를 하나로 평균 — OOF 확률 평균 + fold 모델 목록 합치기 (재학습 없음).

    python tools/merge_seeds.py --stack_dir runs/stack_trv --members lgb lgb_s1 lgb_s2 --out lgb3
결과: oof_lgb3.csv, lgb3.joblib (blend.py에서 멤버 lgb3로 사용). 출력은 집계값만.
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import PCOLS, align_oofs, read_oof, score_fn, write_oof, CLASSES  # noqa: E402
from src.model import cummax_pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack_dir", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    oofs = {m: read_oof(os.path.join(a.stack_dir, f"oof_{m}.csv")) for m in a.members}
    o, Ps, dropped = align_oofs(oofs)
    assert not any(dropped.values()), f"멤버끼리 행이 다릅니다 {dropped}"
    full = oofs[a.members[0]].set_index(["sample_id", "ver"]).loc[list(zip(o.sample_id, o.ver))].reset_index()
    full[PCOLS] = Ps.mean(0)
    write_oof(full, os.path.join(a.stack_dir, f"oof_{a.out}.csv"))
    mems = [joblib.load(os.path.join(a.stack_dir, f"{m}.joblib")) for m in a.members]
    merged = dict(mems[0])
    merged["models"] = [x for mem in mems for x in mem["models"]]
    joblib.dump(merged, os.path.join(a.stack_dir, f"{a.out}.joblib"), compress=3)
    for name, P in [(m, Ps[i]) for i, m in enumerate(a.members)] + [(a.out, Ps.mean(0))]:
        lab = CLASSES[P.argmax(1)]
        print(f"  {name:<10s} raw {score_fn(o, lab):.4f}  cummax {score_fn(o, cummax_pp(o, lab)):.4f}")
    print(f"저장 → {a.stack_dir}/oof_{a.out}.csv, {a.out}.joblib (모델 {len(merged['models'])}개)")


if __name__ == "__main__":
    main()
