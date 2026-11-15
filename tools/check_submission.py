"""제출 전 점검 — 운영진 1차 코드 검증 조건을 흉내 낸다. 집계값만 출력한다 (행·ID·예측값은 출력하지 않음).

    python tools/check_submission.py --input <경로>/valid/valid.csv --image_root <경로>/valid

검사
  1. 행 셔플            : 입력 행 순서를 섞어도 sample_id별 예측이 같아야 한다
  2. 실험 단위 분할      : 실험을 두 묶음으로 나눠 따로 예측해도 같아야 한다
  3. 뒷부분 삭제         : 실험마다 임의 지점 이후 행을 지워도 남은 행 예측이 같아야 한다
  4. 빈 이미지           : 이미지를 같은 크기의 단색 이미지로 바꾸면 예측이 달라져야 한다 (이미지 사용 확인)
입력에 label 열이 있으면 Macro F1(집계)도 출력한다.
"""
import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import KEY, T  # noqa: E402
from src.ensemble import team_image_features  # noqa: E402
from src.evaluation import macro_f1  # noqa: E402
import src.rn18 as rn18  # noqa: E402
import src.team_catboost as TM  # noqa: E402


def _load_test_module():
    spec = importlib.util.spec_from_file_location("submit_test", ROOT / "test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--model", default=str(ROOT / "model.joblib"))
    ap.add_argument("--n_exp", type=int, default=12, help="검사에 쓸 실험 수 (0=전부)")
    ap.add_argument("--blank_value", type=int, default=0, help="빈 이미지 밝기 (0=검정)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tm = _load_test_module()
    bundle = joblib.load(a.model)
    bundle["post"] = dict(bundle["post"])
    is_stack = "members" in bundle
    needs_img = "team" in bundle or (is_stack and any(m["kind"] == "team" or m.get("module") == "src.rn18"
                                                      for m in bundle["members"].values()))
    root = a.image_root or os.path.dirname(os.path.abspath(a.input))
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.input)
    exps = df[KEY].unique()
    if a.n_exp and len(exps) > a.n_exp:
        exps = rng.choice(exps, size=a.n_exp, replace=False)
        df = df[df[KEY].isin(exps)].reset_index(drop=True)
    print(f"입력 {len(df):,} rows, {df[KEY].nunique()} experiments, 이미지 사용 멤버 {'있음' if needs_img else '없음'}")

    def run(frame, img_root):
        rn18.IMAGE_ROOT = img_root
        timg = team_image_features(frame, img_root) if needs_img else None
        return tm.predict_df(bundle, frame, timg)

    base = run(df, root)
    ok = True

    def same(name, p):
        nonlocal ok
        rate = float((p.to_numpy() == base.loc[p.index].to_numpy()).mean())
        ok &= rate == 1.0
        print(f"  {name:<14s} 일치율 {rate:.4f} ({len(p):,} rows) {'OK' if rate == 1.0 else '불일치!'}")

    same("행 셔플", run(df.sample(frac=1, random_state=a.seed).reset_index(drop=True), root))
    half = set(exps[: len(exps) // 2])
    same("실험 분할", pd.concat([run(df[df[KEY].isin(half)].reset_index(drop=True), root),
                              run(df[~df[KEY].isin(half)].reset_index(drop=True), root)]))
    cut = []
    for _, g in df.sort_values([KEY, T]).groupby(KEY, sort=False):
        cut.append(g.iloc[: int(rng.integers(1, len(g) + 1))])
    same("뒷부분 삭제", run(pd.concat(cut).reset_index(drop=True), root))

    with tempfile.TemporaryDirectory() as tmp:
        src = TM.resolve_image_paths(df, [Path(root), Path(root).parent])
        for rel, p in zip(df["image_path"].astype(str), src):
            dst = Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(p) as im:
                Image.new(im.mode, im.size, color=a.blank_value if im.mode in ("L", "I", "F") else
                          (a.blank_value,) * len(im.getbands())).save(dst)
        blank = run(df, tmp)
    changed = float((blank.to_numpy() != base.loc[blank.index].to_numpy()).mean())
    ok &= changed > 0
    print(f"  {'빈 이미지':<14s} 예측 변경 비율 {changed:.4f} {'OK (이미지 사용됨)' if changed > 0 else '변화 없음! 이미지 미사용'}")

    if "label" in df:
        y = df.set_index("sample_id").loc[base.index, "label"].to_numpy()
        print(f"  Macro F1 (검사 실험, 원본 이미지) {macro_f1(y, base.to_numpy()):.4f}  "
              f"(빈 이미지) {macro_f1(y, blank.loc[base.index].to_numpy()):.4f}")
    print("전체 결과:", "통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
