"""스태킹 bundle에서 일부 멤버만 골라 비율·후처리를 고정한 최종 model.joblib을 만든다 (재학습 없음).

    python tools/make_final.py --src model_mix.joblib --members lgb rn18s --weights 0.5 0.5 \
                               --post cummax --out model.joblib
    # 서로 다른 학습 폴더의 멤버 파일을 직접 묶을 때 (--src 없이 이름=경로)
    python tools/make_final.py --members lgb=runs/stack_full/lgb3.joblib rn18s=runs/stack_trv/rn18s3.joblib \
                               --weights 0.5 0.5 --post cummax --out model.joblib
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
    ap.add_argument("--src", default=None, help="blend.py가 만든 bundle. 없으면 --members에 이름=멤버.joblib 경로")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True, help="멤버별 비율 (모든 단계 공통, 합 1)")
    ap.add_argument("--post", default="cummax", choices=["raw", "cummax", "cummax2", "cummax3"])
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--refine23", default=None, help="2↔3 전용 멤버 파일 (예: runs/stack_trv/s23.joblib)")
    ap.add_argument("--refine_alpha", type=float, default=0.5, help="2↔3 전용 모델 반영 비율 (0~1)")
    a = ap.parse_args()
    assert len(a.members) == len(a.weights) and abs(sum(a.weights) - 1) < 1e-6, "비율은 멤버 수만큼, 합 1"
    if a.src:
        b = joblib.load(a.src)
        missing = [m for m in a.members if m not in b["members"]]
        assert not missing, f"bundle에 없는 멤버 {missing} (있는 멤버 {list(b['members'])})"
        b["members"] = {m: b["members"][m] for m in a.members}
    else:
        from src.evaluation import CLASSES
        pairs = [m.split("=", 1) for m in a.members]
        assert all(len(p) == 2 for p in pairs), "--src 없이 쓸 때는 --members 이름=경로"
        b = {"members": {n: joblib.load(path) for n, path in pairs}, "classes": CLASSES, "use_images": False}
        a.members = [n for n, _ in pairs]
        for n, path in pairs:
            print(f"  {n} ← {path} (모델 {len(b['members'][n]['models'])}개)")
    b["blend"] = {m: np.full(len(b["classes"]), w) for m, w in zip(a.members, a.weights)}
    b.pop("refine23", None)
    if a.refine23:
        r = joblib.load(a.refine23)
        assert r["kind"] == "s23", "--refine23에는 s23 멤버 파일을 주세요"
        b["refine23"] = {"models": r["models"], "feats": r["feats"], "alpha": a.refine_alpha}
        print(f"  2↔3 전용 모델 ← {a.refine23} (모델 {len(r['models'])}개, 반영 비율 {a.refine_alpha})")
    b["feats"] = sorted({c for mem in list(b["members"].values()) + ([b["refine23"]] if a.refine23 else [])
                         for c in (mem.get("feats") or [])})
    b["w"] = np.ones(len(b["classes"]))
    b["post"] = {"method": "none" if a.post == "raw" else a.post, "use_w": False, "hmm": None, "gate": "none", "module": None}
    joblib.dump(b, a.out, compress=3)
    print(f"저장 → {a.out} ({os.path.getsize(a.out) / 1e6:.1f}MB)  멤버 {dict(zip(a.members, a.weights))}  후처리 {a.post}")


if __name__ == "__main__":
    main()
