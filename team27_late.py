"""Evaluate #27 with low-confidence confirmation limited to transitions into stages 4–6."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from compare_team_final import (
    FINAL, ID, ROOT, aligned_probs, evaluation_rows, load_final, metrics, read_splits, replay_missing,
)
from src.data import KEY, T
from src.evaluation import CLASSES, macro_f1, score_fn
from src.model import cummax_pp

STRONG = 0.75
MAX_CONFIRM_GAP = 15.0
TITLES = {"base": "#27 원본", "late": "4~6단계 상승만 재확인"}


def confirm(p, t):
    raw = CLASSES[p.argmax(axis=1)]
    # A stage at or above k requires strong probability mass or agreement
    # across two observations. The initial observation keeps its own label.
    supported = raw[:, None] >= CLASSES
    tails = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    previous = np.zeros_like(supported)
    gap = np.diff(t)
    previous[1:] = supported[:-1] & ((gap > 0) & (gap <= MAX_CONFIRM_GAP))[:, None]
    ready = supported & ((tails >= STRONG) | previous)
    ready[:, :3] = supported[:, :3]
    ready[0] = supported[0]
    return (ready * CLASSES).max(axis=1)


def decode(rows, p, w, decoder):
    weighted = np.asarray(p, dtype=float) * w
    weighted /= weighted.sum(axis=1, keepdims=True)
    idx = rows[[KEY, T, ID]].reset_index(drop=True).sort_values([KEY, T, ID])
    pred = np.empty(len(rows), dtype=np.int64)
    for _, g in idx.groupby(KEY, sort=False):
        ix = g.index.to_numpy()
        pred[ix] = np.maximum.accumulate(decoder(weighted[ix], g[T].to_numpy(float)))
    return pred


def confirmed_labels(rows, p, w):
    return decode(rows, p, w, confirm)


POSTS = {"late": confirmed_labels}


def group_checks(rows, preds):
    summaries = []
    for split in ("train", "validation"):
        use = rows["split"].eq(split) & rows["ver"].eq("k1o0")
        part = rows.loc[use]
        truth = part["y"].to_numpy()
        label_set = np.unique(truth)
        for method, group in part.groupby("tr_method", dropna=False, sort=False):
            ix = group.index.to_numpy()
            actual = group["y"].to_numpy()
            labels = np.intersect1d(label_set, actual)
            summaries.append({
                "평가": split, "tr_method": method, "실험": group["grp"].nunique(), "단계 수": len(labels),
                **{TITLES[name]: macro_f1(actual, pred[ix], labels=labels) for name, pred in preds.items()},
            })
        per_exp = []
        for _, exp in part.groupby("grp", sort=False):
            actual = exp["y"].to_numpy()
            labels = np.unique(actual)
            per_exp.append([
                macro_f1(actual, pred[exp.index.to_numpy()], labels=labels) for pred in preds.values()
            ])
        values = np.asarray(per_exp)
        print(f"{split} 실험별 F1 하위 10% 지점 · " + " / ".join(
            f"{TITLES[name]} {value:.5f}" for name, value in zip(preds, np.quantile(values, 0.10, axis=0))
        ))
        for i, name in enumerate(tuple(preds)[1:], start=1):
            delta = values[:, i] - values[:, 0]
            print(f"  {TITLES[name]} · 개선 {(delta > 1e-9).sum()} / 악화 {(delta < -1e-9).sum()}"
                  f" / 동일 {(np.abs(delta) <= 1e-9).sum()}개 실험")
    print(pd.DataFrame(summaries).to_string(index=False, float_format="{:.5f}".format))


def compare(args, bundle):
    frames = read_splits(args.data_dir)
    rows = evaluation_rows(frames, args.stack_dir)
    members = bundle["members"]
    probs, missing = {}, {}
    for name in FINAL:
        path = args.stack_dir / f"oof_{name}.csv"
        if path.is_file():
            probs[name] = aligned_probs(rows, path)
        else:
            missing[name] = members[name]
    if missing:
        print(f"저장 모델로 누락된 확률만 복원 · {', '.join(missing)}", flush=True)
        probs.update(replay_missing(rows, frames, missing, args))
    p = sum(FINAL[name] * probs[name] for name in FINAL)
    preds = {"base": cummax_pp(rows, CLASSES[p.argmax(axis=1)])}
    preds.update({name: apply(rows, p, np.ones(len(CLASSES))) for name, apply in POSTS.items()})
    val5 = rows["split"].eq("validation").to_numpy() & rows["ver"].eq("k1o0").to_numpy()
    y = rows["y"].to_numpy()
    table = []
    for name, pred in preds.items():
        row = metrics(rows, pred, TITLES[name])
        row["Val 2→3"] = int((val5 & (y == 2) & (pred == 3)).sum())
        table.append(row)
    print("동일 실험 폴드 · 원본 5초 및 간격 가중 Macro F1")
    print(pd.DataFrame(table).to_string(index=False, float_format="{:.5f}".format))
    if any(abs(table[0][k] - v) > 0.00005 for k, v in (("OOF 5초", 0.90310), ("전체 Val 5초", 0.91340))):
        print("기준 점수 확인 필요 · #27 기록은 OOF 0.90310 / Val 0.91340")
    else:
        print("#27 서버 검증 기록과 기준 점수 일치")
    print("\n실험 유형·개별 실험 진단 · 해당 실험/유형에 실제 등장한 단계만 평균")
    group_checks(rows, preds)
    print("\n#27 Public 0.8917411656 · 확장안 Public 미측정")


def make_submission(args, base):
    mode = args.make_submission
    out = args.out_dir.resolve()
    data = args.data_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = {**base, "post": {**base["post"], "method": mode, "module": "team27_late"}}
    with TemporaryDirectory(dir=out) as temp:
        model = Path(temp) / f"model_team27_{mode}.joblib"
        csv = Path(temp) / f"submission_team27_{mode}.csv"
        sample = data / "sample_submission.csv"
        joblib.dump(bundle, model, compress=3)
        flags = ["--input", str(data / "test/test.csv"), "--image_root", str(data / "test"),
                 "--model", str(model)]
        subprocess.run([sys.executable, "-u", "test.py", *flags, "--sample_submission", str(sample),
                        "--output", str(csv), "--skip_check"], cwd=ROOT, check=True)
        result = pd.read_csv(csv, dtype={ID: str})
        expected = pd.read_csv(sample, dtype={ID: str})[ID]
        if (list(result.columns) != [ID, "label"] or not result[ID].is_unique
                or not result[ID].equals(expected) or not pd.api.types.is_integer_dtype(result["label"])
                or not result["label"].isin(CLASSES).all()):
            raise ValueError("생성된 CSV의 sample_id·순서·라벨이 제출 양식과 다릅니다.")
        subprocess.run([sys.executable, "-u", "tools/check_submission.py", *flags, "--n_exp", "0"],
                       cwd=ROOT, check=True)
        model.replace(out / model.name)
        csv.replace(out / csv.name)
    print(f"제출 형식과 네 가지 점검 통과 · {out / csv.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path.home() / "data/battery_data_light")
    parser.add_argument("--stack_dir", type=Path, default=Path.home() / "unid-data-competition/runs/stack_trv")
    parser.add_argument("--model", type=Path, default=ROOT / "model.joblib")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "runs/team27_late")
    parser.add_argument("--make_submission", choices=POSTS, help="지정한 후보의 Test CSV 생성 및 점검")
    args = parser.parse_args()
    base = load_final(args.model)
    if args.make_submission:
        make_submission(args, base)
    else:
        compare(args, base)


if __name__ == "__main__":
    main()
