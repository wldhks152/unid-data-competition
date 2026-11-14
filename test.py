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
from src.ensemble import blend_proba, team_image_features
from src.features import make_features
from src.images import add_image_features
from src.model import predict_labels


def predict_df(bundle, df, timg=None):
    """df 행만으로 예측. timg: 팀원 CatBoost용 이미지 통계 (행별 독립 계산, 앙상블일 때만)."""
    F = make_features(df).reset_index(drop=True)
    missing = [c for c in bundle["feats"] if c not in F.columns]
    if missing:
        print(f"  [경고] 피처 {len(missing)}개 없음 → NaN으로 채움 (예: {missing[:3]})")
        F = F.assign(**{c: np.nan for c in missing})
    P = blend_proba(bundle, F, df, timg)
    pred, _ = predict_labels(bundle, F, P=P)
    return pd.Series(pred, index=F.sample_id.values)


def causality_check(bundle, df, pred_full, timg=None, n_exp=5, seed=0):
    """입력의 일부 실험을 임의 지점에서 잘라 다시 예측 → 남은 행 예측이 전체 예측과 같아야 한다."""
    rng = np.random.default_rng(seed)
    exps = rng.choice(df[KEY].unique(), size=min(n_exp, df[KEY].nunique()), replace=False)
    for e in exps:
        g = df[df[KEY] == e].sort_values(T)
        cut = int(rng.integers(1, len(g) + 1))
        p = predict_df(bundle, g.iloc[:cut], timg)
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
    # 진단용 덮어쓰기 (재학습 없이 같은 model.joblib로 변형 제출 파일을 만든다)
    ap.add_argument("--alpha", type=float, default=None, help="앙상블 LightGBM 비율 덮어쓰기 (1=LightGBM만, 0=CatBoost만)")
    ap.add_argument("--post", choices=["none", "cummax", "cummax2", "cummax3", "cummax4"], default=None, help="후처리 덮어쓰기")
    ap.add_argument("--use_w", type=int, choices=[0, 1], default=None, help="클래스 배율 w 사용 여부 덮어쓰기")
    ap.add_argument("--gate", choices=["none", "12", "1234"], default=None, help="단계 전환 규칙 게이트 (src/model.py rule_gate)")
    ap.add_argument("--alpha_split", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="클래스별 LightGBM 비율: 1~3단계 LO, 4~6단계 HI")
    ap.add_argument("--w", type=float, nargs="+", default=None, help="클래스 배율 직접 지정 (6개)")
    ap.add_argument("--w_power", type=float, default=None, help="클래스 배율을 w**p로 완화 (0=배율 없음, 1=그대로)")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    bundle["post"] = dict(bundle["post"])
    if a.post is not None:
        bundle["post"]["method"] = a.post
    if a.use_w is not None:
        bundle["post"]["use_w"] = bool(a.use_w)
    if a.gate is not None:
        bundle["post"]["gate"] = a.gate
    if a.alpha_split is not None:
        if "team" not in bundle:
            sys.exit("[오류] CatBoost가 없는 model.joblib입니다.")
        lo, hi = a.alpha_split
        bundle["alpha"] = np.array([lo if c <= 3 else hi for c in bundle["classes"]], dtype=float)
    if a.w is not None:
        assert len(a.w) == len(bundle["classes"]), "--w는 클래스 수만큼"
        bundle["w"] = np.asarray(a.w, dtype=float)
        bundle["post"]["use_w"] = True
    if a.w_power is not None:
        bundle["w"] = np.asarray(bundle["w"], dtype=float) ** a.w_power
        print(f"  w → w**{a.w_power}: {np.round(bundle['w'], 2).tolist()}")
    if a.alpha is not None:
        if a.alpha >= 1 and "team" in bundle:
            bundle.pop("team")
        elif "team" in bundle:
            bundle["alpha"] = a.alpha
        elif a.alpha < 1:
            sys.exit("[오류] 이 model.joblib에는 CatBoost가 없어 --alpha < 1을 쓸 수 없습니다.")
    is_stack = "members" in bundle
    needs_team = "team" in bundle or (is_stack and any(m["kind"] == "team" or m.get("module") == "src.rn18"
                                                       for m in bundle["members"].values()))
    ens = ("스태킹(" + ", ".join(f"{n}={np.round(np.asarray(w), 2).tolist()}" for n, w in bundle["blend"].items()) + ")") if is_stack else f"앙상블(LightGBM 비율 {np.round(bundle['alpha'], 2).tolist()})" if "team" in bundle else "LightGBM 단독"
    print(f"  모델: {a.model}  {ens}, 후처리={bundle['post']['method']}, w={bundle['post']['use_w']}, "
          f"gate={bundle['post'].get('gate', 'none')}, "
          f"이미지={bundle.get('use_images', False)}")
    df = pd.read_csv(a.input)
    print(f"  입력: {a.input}  {len(df):,} rows, {df[KEY].nunique()} experiments")
    root = a.image_root or os.path.dirname(os.path.abspath(a.input))
    import src.rn18 as rn18
    rn18.IMAGE_ROOT = root
    if bundle.get("use_images", False):
        df = add_image_features(df, root)
    timg = team_image_features(df, root) if needs_team else None

    pred = predict_df(bundle, df, timg)
    if not a.skip_check:
        causality_check(bundle, df, pred, timg)

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
