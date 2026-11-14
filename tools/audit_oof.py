"""Audit experiment-grouped OOF predictions; does not prove training independence.

Usage: python tools/audit_oof.py --oof runs/model_oof.csv --report runs/audit.json
Optional --fold-manifest manifest.csv must contain grp,fold (one row/experiment).
Probability blocks such as p1..p6, q1..q6 or lgb_1..lgb_6 are detected.
Only pandas, numpy and sklearn are needed; no model training or GPU is used.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [1, 2, 3, 4, 5, 6]
VERSION_WEIGHTS = {"k1o0": 9, "k2o0": 1, "k3o0": 1}


def weighted_macro_f1(frame, pred):
    """One objective for tuning/reporting: 9:1:1 of version macro F1, labels 1..6.

    Missing versions are errors, rather than silently changing the objective.
    Calling this on selected OOF parameters gives a development score, not an
    independent or nested-CV performance estimate.
    """
    if len(pred) != len(frame):
        raise ValueError("Prediction length differs from OOF rows")
    versions = set(frame["ver"])
    if versions != set(VERSION_WEIGHTS):
        raise ValueError(f"Expected versions {list(VERSION_WEIGHTS)}, got {sorted(map(str, versions))}")
    pred = np.asarray(pred)
    scores = {v: float(f1_score(frame.loc[frame.ver.eq(v), "y"], pred[frame.ver.eq(v).to_numpy()],
                               labels=LABELS, average="macro", zero_division=0))
              for v in VERSION_WEIGHTS}
    return sum(VERSION_WEIGHTS[v] * scores[v] for v in scores) / 11, scores


def audit_oof(frame, manifest=None):
    errors = []
    result = {"ok": False, "rows": len(frame), "errors": errors,
              "scope": "OOF structure and development scores only; training separation, causality, and nested CV require separate verification."}
    required = ["sample_id", "experiment_id", "ver", "fold", "y"]
    missing = [c for c in required if c not in frame]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return result
    if frame.empty:
        errors.append("OOF is empty")
        return result
    df = frame.copy()
    if df[required].isna().any().any():
        errors.append("Missing sample/experiment/version/fold/label value")
        return result
    if df.duplicated(["sample_id", "ver"]).any():
        errors.append("Duplicate (sample_id, ver) rows")
    if set(df.ver) != set(VERSION_WEIGHTS):
        errors.append(f"Expected all three versions {list(VERSION_WEIGHTS)}")
    if not df.y.isin(LABELS).all():
        errors.append("Labels must be integers 1..6")
    fold = pd.to_numeric(df.fold, errors="coerce")
    if not np.isfinite(fold).all() or (fold % 1 != 0).any() or (fold < 0).any():
        errors.append("Fold IDs must be nonnegative integers")
    elif fold.nunique() < 2:
        errors.append("OOF must contain at least two folds")
    else:
        df["fold"] = fold.astype(int)

    # subsample() appends the exact version suffix to experiment_id. Strip only
    # that suffix so augmented copies of the same original experiment stay tied.
    def original_id(row):
        exp, suffix = str(row.experiment_id), "__" + str(row.ver)
        return exp[:-len(suffix)] if exp.endswith(suffix) else exp

    inferred = df.apply(original_id, axis=1)
    if "grp" in df:
        if df.grp.isna().any() or not df.grp.astype(str).eq(inferred).all():
            errors.append("grp disagrees with experiment_id/version suffix")
        df["_group"] = df.grp.astype(str)
    else:
        df["_group"] = inferred
    leaked = df.groupby("_group").fold.nunique()
    if (leaked > 1).any():
        errors.append(f"Original experiments span multiple folds: {leaked[leaked > 1].index.tolist()[:5]}")
    if (df.groupby("sample_id").y.nunique() > 1).any():
        errors.append("A sample has inconsistent labels across versions")
    if (df.groupby("sample_id")._group.nunique() > 1).any():
        errors.append("A sample belongs to different original experiments")
    groups = {v: set(df.loc[df.ver.eq(v), "_group"]) for v in VERSION_WEIGHTS}
    if len({frozenset(s) for s in groups.values()}) > 1:
        errors.append("Version subsets cover different original experiments")
    base = df[df.ver.eq("k1o0")].set_index("sample_id")["_group"].to_dict()
    for row in df[~df.ver.eq("k1o0")].itertuples():
        if row.sample_id not in base:
            errors.append("Downsample version contains sample absent from k1o0")
            break
    if "relative_time_s" in df:
        time = pd.to_numeric(df.relative_time_s, errors="coerce")
        if not np.isfinite(time).all() or (time < 0).any():
            errors.append("relative_time_s must be finite and nonnegative")
        if df.duplicated(["_group", "ver", "relative_time_s"]).any():
            errors.append("Duplicate timestamp within experiment/version")

    if manifest is not None:
        if not {"grp", "fold"}.issubset(manifest.columns):
            errors.append("Manifest requires grp,fold")
        elif manifest[["grp", "fold"]].isna().any().any() or manifest.grp.astype(str).duplicated().any():
            errors.append("Manifest groups must be unique and nonmissing; folds nonmissing")
        else:
            lookup = manifest.assign(grp=manifest.grp.astype(str)).set_index("grp").fold
            expected = df._group.map(lookup)
            if expected.isna().any() or not pd.to_numeric(expected, errors="coerce").eq(fold).all():
                errors.append("OOF fold assignment disagrees with manifest")

    blocks = {}
    for col in df.columns:
        match = re.fullmatch(r"(.+?)([1-6])", str(col))
        if match:
            blocks.setdefault(match[1], {})[int(match[2])] = col
    # Recognize standard p/q/r blocks even when a class column is missing.
    for prefix, cols in blocks.items():
        if prefix in {"p", "q", "r"} and set(cols) != set(LABELS):
            errors.append(f"Incomplete probability block {prefix}: expected classes 1..6")
    complete = {prefix: [cols[k] for k in LABELS] for prefix, cols in blocks.items() if set(cols) == set(LABELS)}
    if not complete:
        errors.append("No complete probability block (e.g. p1..p6)")
    for prefix, cols in complete.items():
        p = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            errors.append(f"Invalid probability values in {prefix}")
        elif not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-4):
            errors.append(f"Probability rows do not sum to 1 in {prefix}")
    result.update(experiments=int(df._group.nunique()), folds=int(df.fold.nunique()),
                  probability_blocks=list(complete), manifest_checked=manifest is not None)
    if errors:
        return result
    benchmarks = {}
    for prefix, cols in complete.items():
        pred = np.asarray(LABELS)[df[cols].to_numpy(float).argmax(axis=1)]
        score, by_version = weighted_macro_f1(df, pred)
        by_fold = {}
        for f in sorted(df.fold.unique()):
            mask = df.fold.eq(f).to_numpy()
            fs, _ = weighted_macro_f1(df.loc[mask], pred[mask])
            by_fold[str(f)] = fs
        benchmarks[prefix] = {"weighted_macro_f1": score, "by_version": by_version,
                               "by_fold": by_fold, "fold_std": float(np.std(list(by_fold.values())))}
    result.update(ok=True, raw_oof_development_scores=benchmarks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fold-manifest")
    args = parser.parse_args()
    frame = pd.read_csv(args.oof, dtype={"sample_id": str, "experiment_id": str, "grp": str, "ver": str})
    manifest = pd.read_csv(args.fold_manifest, dtype={"grp": str}) if args.fold_manifest else None
    report = audit_oof(frame, manifest)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
