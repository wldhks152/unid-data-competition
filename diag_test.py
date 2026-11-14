"""진단 — Test 정답 없이 여러 제출 파일을 실험별로 비교한다.
    python diag_test.py --input <경로>/test/test.csv \
        --subs lgb=sub_lgb.csv cat=sub_cat.csv ens=sub_ens.csv \
        [--public lgb=0.8274 cat=0.8828 ens=0.8144] [--out diag]

출력
  1) 실험별 조건 · 측정 간격 · 규칙 사건 시각 (t1 ≥ 45°C, 전압 사건)
  2) 제출별 단계 진행 (각 단계가 처음 나온 시각) · 규칙과 어긋나는 행 수 · 단계 역행 수
  3) 제출 쌍별로 예측이 다른 행 수 (실험별)
  4) Public 점수가 같은 두 제출이 다른 행을 가진 실험 → Public에 안 들어간 실험일 가능성이 큼
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.data import KEY, T
from src.features import make_features


def parse_pairs(items, cast=str):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = cast(v)
    return out


def rule_flags(F):
    t45 = F["ev_t1_above45"].to_numpy() > 0
    heat = F["tr_method"].to_numpy() == 1
    vev = (F["ev_v_below2.45"].to_numpy() > 0) | np.where(
        heat, F["ev_v_drop0.78"].to_numpy() > 0, F["ev_v_drop1.0"].to_numpy() > 0)
    return t45, vev


def first_time(t, mask):
    return float(t[mask][0]) if mask.any() else np.nan


def stage_path(t, y):
    """각 단계가 처음 나온 시각: '1@0 2@455 3@900 ...'"""
    seen, parts = set(), []
    for ti, yi in zip(t, y):
        if yi not in seen:
            seen.add(yi)
            parts.append(f"{yi}@{ti:.0f}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--subs", nargs="+", required=True, help="이름=제출CSV 경로")
    ap.add_argument("--public", nargs="*", default=None, help="이름=Public 점수 (선택)")
    ap.add_argument("--out", default="diag", help="CSV 저장 폴더")
    a = ap.parse_args()
    subs = parse_pairs(a.subs)
    pub = parse_pairs(a.public, float)
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.input)
    F = make_features(df).reset_index(drop=True)  # 실험·시간순
    for name, path in subs.items():
        s = pd.read_csv(path).set_index("sample_id")["label"]
        miss = (~F.sample_id.isin(s.index)).sum()
        assert miss == 0, f"{name}: test 행 {miss}개가 제출에 없음"
        F[f"y_{name}"] = s.loc[F.sample_id].to_numpy().astype(int)
    t45, vev = rule_flags(F)
    F["_t45"], F["_vev"] = t45, vev
    names = list(subs)

    rows = []
    for e, g in F.groupby(KEY, sort=False):
        t = g[T].to_numpy(float)
        r = {"exp": e, "rows": len(g), "dt": float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
             "tr": int(g.tr_method.iloc[0]), "cls": int(g.bat_class.iloc[0]), "cc": g.bat_cc.iloc[0],
             "soc": int(g.soc_status.iloc[0]), "t_end": t[-1],
             "t45@": first_time(t, g._t45.to_numpy()), "Vev@": first_time(t, g._vev.to_numpy())}
        for n in names:
            y = g[f"y_{n}"].to_numpy()
            r[f"{n}|path"] = stage_path(t, y)
            r[f"{n}|down"] = int((np.diff(y) < 0).sum())
            r[f"{n}|rule12"] = int(((y >= 2) & ~g._t45.to_numpy()).sum() + ((y == 1) & g._t45.to_numpy()).sum())
            r[f"{n}|rule34"] = int(((y >= 4) & ~g._vev.to_numpy()).sum() + ((y <= 3) & g._vev.to_numpy()).sum())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                r[f"diff:{n1}-{n2}"] = int((g[f"y_{n1}"] != g[f"y_{n2}"]).sum())
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "per_experiment.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 80)
    print("=== 1) 실험 조건과 규칙 사건 시각 (초) ===")
    print(R[["exp", "rows", "dt", "tr", "cls", "cc", "soc", "t_end", "t45@", "Vev@"]].to_string(index=False))

    print("\n=== 2) 제출별 단계 진행 (단계@처음 나온 시각) / 규칙 어긋남 행 수 / 역행 수 ===")
    for _, r in R.iterrows():
        print(f"\n[{r['exp']}] {r['rows']}행 {r['dt']:.0f}s  tr={r['tr']} cls={r['cls']} cc={r['cc']}  t45@{r['t45@']}  Vev@{r['Vev@']}")
        for n in names:
            print(f"   {n:<10s} {r[f'{n}|path']:<45s} rule12={r[f'{n}|rule12']:<4d} rule34={r[f'{n}|rule34']:<4d} down={r[f'{n}|down']}")

    print("\n=== 3) 제출 쌍별 예측이 다른 행 수 ===")
    dcols = [c for c in R.columns if c.startswith("diff:")]
    if dcols:
        print(R[["exp"] + dcols].to_string(index=False))
        print("  합계:", R[dcols].sum().to_dict())

    print("\n=== 전체 요약 ===")
    for n in names:
        y = F[f"y_{n}"]
        print(f"  {n:<10s} 분포 {y.value_counts().sort_index().to_dict()}  "
              f"rule12 어긋남 {R[f'{n}|rule12'].sum()}  rule34 어긋남 {R[f'{n}|rule34'].sum()}"
              + (f"  Public {pub[n]}" if n in pub else ""))

    if len(pub) >= 2:
        print("\n=== 4) Public 점수가 같은 제출 쌍 → 다른 행이 있는 실험은 Public 밖일 가능성 ===")
        found = False
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                if n1 in pub and n2 in pub and abs(pub[n1] - pub[n2]) < 1e-9:
                    found = True
                    c = f"diff:{n1}-{n2}"
                    print(f"  {n1} vs {n2}: Public 동일, 다른 행이 있는 실험 = {R.loc[R[c] > 0, 'exp'].tolist()}")
        if not found:
            print("  (Public 점수가 같은 쌍 없음)")
    print(f"\n저장: {os.path.join(a.out, 'per_experiment.csv')}")


if __name__ == "__main__":
    main()
