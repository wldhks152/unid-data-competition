"""스태킹 bundle에서 일부 멤버만 골라 비율·후처리를 고정한 최종 model.joblib을 만든다 (재학습 없음).

    python tools/make_final.py --src model_mix.joblib --members lgb rn18s --weights 0.5 0.5 \
                               --post cummax --out model.joblib
test.py는 이 파일만 읽으면 되고, 덮어쓰기 옵션 없이 같은 예측을 낸다.
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True, help="멤버별 비율 (모든 단계 공통, 합 1)")
    ap.add_argument("--post", default="cummax", choices=["raw", "cummax", "cummax2", "cummax3"])
    ap.add_argument("--out", default="model.joblib")
    a = ap.parse_args()
    assert len(a.members) == len(a.weights) and abs(sum(a.weights) - 1) < 1e-6, "비율은 멤버 수만큼, 합 1"
    b = joblib.load(a.src)
    missing = [m for m in a.members if m not in b["members"]]
    assert not missing, f"bundle에 없는 멤버 {missing} (있는 멤버 {list(b['members'])})"
    b["members"] = {m: b["members"][m] for m in a.members}
    b["blend"] = {m: np.full(len(b["classes"]), w) for m, w in zip(a.members, a.weights)}
    b["feats"] = sorted({c for mem in b["members"].values() for c in (mem.get("feats") or [])})
    b["w"] = np.ones(len(b["classes"]))
    b["post"] = {"method": "none" if a.post == "raw" else a.post, "use_w": False, "hmm": None, "gate": "none", "module": None}
    joblib.dump(b, a.out, compress=3)
    print(f"저장 → {a.out} ({os.path.getsize(a.out) / 1e6:.1f}MB)  멤버 {dict(zip(a.members, a.weights))}  후처리 {a.post}")


if __name__ == "__main__":
    main()
