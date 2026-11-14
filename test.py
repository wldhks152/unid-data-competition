"""추론 — 입력 CSV에 들어 있는 행만으로 예측하고 제출 파일을 만든다 (재학습 없음).
    python test.py --input <경로>/test/test.csv --image_root <경로>/test \
                   --model model.joblib --output submission.csv \
                   --sample_submission <경로>/sample_submission.csv

* 피처는 같은 실험의 현재·과거 행만 사용 → 운영진이 미래 행을 지운 CSV로 재실행해도
  남은 행의 예측은 전체 CSV로 예측했을 때와 같다 (실행 시 자동 검사).
* 입력이 잘린 경우: sample_submission 순서를 따르되, 입력에 있는 행만 출력한다.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

from src.data import KEY, T
from src.features import make_features
from src.images import add_image_features
from src.model import predict_labels


def predict_df(bundle, df):
    F = make_features(df)
    missing = [c for c in bundle["feats"] if c not in F.columns]
    if missing:
        print(f"  [경고] 피처 {len(missing)}개 없음 → NaN으로 채움 (예: {missing[:3]})")
        F = F.assign(**{c: np.nan for c in missing})
    pred, _ = predict_labels(bundle, F)
    return pd.Series(pred, index=F.sample_id.values)


def causality_check(bundle, df, pred_full, n_exp=5, seed=0):
    """입력의 일부 실험을 임의 지점에서 잘라 다시 예측 → 남은 행 예측이 전체 예측과 같아야 한다."""
    rng = np.random.default_rng(seed)
    exps = rng.choice(df[KEY].unique(), size=min(n_exp, df[KEY].nunique()), replace=False)
    for e in exps:
        g = df[df[KEY] == e].sort_values(T)
        cut = int(rng.integers(1, len(g) + 1))
        p = predict_df(bundle, g.iloc[:cut])
        if not (p.to_numpy() == pred_full.loc[p.index].to_numpy()).all():
            sys.exit(f"[누수!] 실험 {e}: 미래 행을 빼면 예측이 달라집니다.")
    print(f"  누수 검사 OK — {len(exps)}개 실험을 잘라도 남은 행의 예측 동일")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="test.csv 경로")
    ap.add_argument("--image_root", default=None, help="images/ 가 들어 있는 폴더 (기본: input CSV 폴더)")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--sample_submission", default=None)
    ap.add_argument("--skip_check", action="store_true", help="누수 자가검사 생략")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    print(f"  모델: {a.model}  후처리={bundle['post']['method']}, w={bundle['post']['use_w']}, "
          f"이미지={bundle.get('use_images', False)}")
    df = pd.read_csv(a.input)
    print(f"  입력: {a.input}  {len(df):,} rows, {df[KEY].nunique()} experiments")
    if bundle.get("use_images", False):
        root = a.image_root or os.path.dirname(os.path.abspath(a.input))
        df = add_image_features(df, root)

    pred = predict_df(bundle, df)
    if not a.skip_check:
        causality_check(bundle, df, pred)

    if a.sample_submission:
        ss = pd.read_csv(a.sample_submission)
        sub = ss.loc[ss.sample_id.isin(pred.index), ["sample_id"]].copy()
        extra = len(pred) - len(sub)
        if extra:
            print(f"  [경고] sample_submission에 없는 입력 행 {extra}개는 제외")
        if len(sub) < len(ss):
            print(f"  [안내] 입력에 없는 {len(ss) - len(sub)}행은 출력하지 않음 (잘린 입력)")
    else:
        sub = df[["sample_id"]].copy()
    sub["label"] = pred.loc[sub.sample_id].to_numpy().astype(int)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    sub.to_csv(a.output, index=False)
    print(f"  저장: {a.output}  ({len(sub):,} rows)")
    print(f"  예측 분포: {sub.label.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
