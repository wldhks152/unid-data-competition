"""Compare the team's released final ensemble with one ResNet18 extension on the GPU server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.data import EVAL_VERSIONS, KEY, T, subsample
from src.evaluation import CLASSES, OOF_COLS, PCOLS, experiment_folds, macro_f1, read_oof, score_fn
from src.model import cummax_pp

ROOT = Path(__file__).resolve().parent
ID = "sample_id"
ROW_KEY = [ID, "ver"]
MODEL_BLOB = "89bff3ac7fe9bb16338e11f7f8d6f756cbe9aa7a"
FINAL = {"lgb": 0.40, "dino": 0.40, "seq_gru": 0.20}
HYBRID = {"lgb": 0.40, "dino": 0.30, "seq_gru": 0.20, "rn18s": 0.10}


def load_final(path):
    with path.open("rb") as source:
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_BLOB:
        raise ValueError(f"{path}가 GitHub의 팀 최종 model.joblib과 다릅니다. 새 diag/public-drop 폴더를 사용하세요.")
    bundle = joblib.load(path)
    members = bundle["members"]
    post = bundle["post"]
    if (set(members) != set(FINAL) or not np.array_equal(bundle["classes"], CLASSES)
            or any(not np.allclose(bundle["blend"][m], w) for m, w in FINAL.items())
            or any(len(m["models"]) != 6 for m in members.values())
            or post["method"] != "cummax" or post["use_w"] or post.get("gate") not in (None, "none")
            or bundle.get("refine23") or members["dino"].get("embedder") != "dino"):
        raise ValueError("저장 모델의 멤버·가중치·후처리가 예상한 팀 최종 설정과 다릅니다.")
    return bundle


def find_rn18(args):
    candidates = [
        args.stack_dir / "rn18s.joblib",
        ROOT.parent / "uni_d_public_drop/runs/replay_lgb_v2_cm.J8tFPZ/model_65_35.joblib",
    ]
    path = args.rn18_model or next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    bundle = joblib.load(path)
    member = bundle["members"]["rn18s"] if "members" in bundle else bundle
    if (member.get("module") != "src.rn18" or member.get("embedder", "rn18") != "rn18"
            or len(member["models"]) != 6 or not all(m.get("sensors") for m in member["models"])):
        raise ValueError(f"{path}에 Train 5-fold + 홀드아웃용 모델이 담긴 rn18s 멤버가 필요합니다.")
    return member


def read_splits(data_dir):
    frames = {}
    for name in ("train", "validation"):
        path = data_dir / name / f"{name}.csv"
        rows = pd.read_csv(path, dtype={ID: str, KEY: str})
        if (rows.empty or not rows[ID].is_unique or rows[[ID, KEY, T]].isna().any().any()
                or not np.isfinite(rows[T]).all() or not rows["label"].isin(CLASSES).all()):
            raise ValueError(f"{path}의 ID·시각·라벨을 확인하세요.")
        frames[name] = rows
    train, val = frames.values()
    if not set(train[KEY]).isdisjoint(val[KEY]) or not set(train[ID]).isdisjoint(val[ID]):
        raise ValueError("Train과 Validation은 실험과 행이 서로 겹치면 안 됩니다.")
    return frames


def evaluation_rows(frames, stack_dir):
    splits = experiment_folds(frames["train"])
    fold_file = stack_dir / "folds.json"
    if fold_file.is_file() and [set(f) for f in json.loads(fold_file.read_text())] != [va for _, va in splits]:
        raise ValueError(f"{fold_file}을 Train 전용 실험 폴드와 대조하세요.")
    fold_by_group = {group: fold for fold, (_, valid) in enumerate(splits) for group in valid}
    fold_by_group.update(dict.fromkeys(frames["validation"][KEY], len(splits)))
    versions = []
    for k, offset in EVAL_VERSIONS:
        for split, raw in frames.items():
            part = subsample(raw, k, offset)
            part["fold"] = part["grp"].map(fold_by_group)
            part["split"] = split
            part["y"] = part["label"]
            versions.append(part)
    return pd.concat(versions, ignore_index=True)


def aligned_probs(rows, path):
    actual = read_oof(path).set_index(ROW_KEY)
    expected = rows[OOF_COLS].set_index(ROW_KEY)
    if len(actual) != len(expected) or not expected.index.isin(actual.index).all():
        raise ValueError(f"{path}가 공식 Train과 Validation의 전체 평가 행을 포함해야 합니다.")
    actual = actual.loc[expected.index]
    metadata = [c for c in OOF_COLS if c not in ROW_KEY]
    pd.testing.assert_frame_equal(actual[metadata], expected[metadata], check_dtype=False)
    return actual[PCOLS].to_numpy()


def replay_missing(rows, frames, members, args):
    from importlib import import_module

    from src import dino, rn18
    from src.ensemble import team_image_features
    from src.features import make_features
    from src.members import predict_member

    vis_names = members.keys() & {"dino", "rn18s"}
    stats, image_embeddings = None, {}
    if vis_names:
        stats = pd.concat([team_image_features(raw, args.data_dir / split) for split, raw in frames.items()])
        for name in sorted(vis_names):
            encoder = dino if name == "dino" else rn18
            image_embeddings[name] = pd.concat([
                encoder.embeddings(raw, args.data_dir / split) for split, raw in frames.items()
            ])
    rebuilt = {name: np.empty((len(rows), len(CLASSES))) for name in members}
    for (version, fold), raw in rows.groupby(["ver", "fold"], sort=False):
        features = make_features(raw) if "lgb" in members else raw
        position = pd.Series(raw.index, index=raw[ID]).loc[features[ID]].to_numpy()
        for name, member in members.items():
            saved = member["models"][int(fold)]
            if name in vis_names:
                probs = rn18.proba(saved, raw, stats, image_embeddings[name], features[ID].to_numpy())
            elif name == "lgb":
                probs = predict_member(member["kind"], saved, features, member["feats"])
            else:
                module = import_module(member["module"])
                probs = module.predict_proba({**member, "models": [saved]}, features, raw)
            rebuilt[name][position] = probs
        print(f"  저장 가중치로 {version} · fold {fold} 예측 복원", flush=True)
    for name, pred in rebuilt.items():
        if not np.isfinite(pred).all() or (pred < 0).any() or not np.allclose(pred.sum(axis=1), 1, atol=1e-3):
            raise ValueError(f"{name} 예측 확률이 올바르지 않습니다.")
    return rebuilt


def metrics(rows, pred, name):
    y = rows["y"].to_numpy()
    held = rows["split"].eq("validation").to_numpy()
    native = rows["ver"].eq("k1o0").to_numpy()
    fit5, val5 = native & ~held, native & held
    oof = macro_f1(y[fit5], pred[fit5])
    val = macro_f1(y[val5], pred[val5])
    groups_with_stage5 = rows.loc[val5 & (y == 5), "grp"].value_counts()
    rest = val5 & rows["grp"].ne(groups_with_stage5.index[0]).to_numpy() if len(groups_with_stage5) else val5
    folds = rows["fold"].to_numpy()
    return {
        "모델": name, "OOF 5초": oof, "전체 Val 5초": val, "두 점수 최솟값": min(oof, val),
        "최저 fold": min(macro_f1(y[m], pred[m]) for m in (fit5 & (folds == f) for f in np.unique(folds[fit5]))),
        "긴 실험 제외 Val": macro_f1(y[rest], pred[rest]) if rest.any() else None,
        "OOF 9:1:1": score_fn(rows.loc[~held], pred[~held]),
        "Val 9:1:1": score_fn(rows.loc[held], pred[held]),
        "Val 3→2": int((val5 & (y == 3) & (pred == 2)).sum()),
        "Val 5→4": int((val5 & (y == 5) & (pred == 4)).sum()),
        "Val 4→5": int((val5 & (y == 4) & (pred == 5)).sum()),
    }


def compare(args, bundle):
    frames = read_splits(args.data_dir)
    rows = evaluation_rows(frames, args.stack_dir)
    probabilities, missing = {}, {}
    members = dict(bundle["members"])
    for name in HYBRID:
        path = args.stack_dir / f"oof_{name}.csv"
        if name == "rn18s" and not path.is_file():
            member = find_rn18(args)
            if member is None:
                print("rn18s OOF/가중치가 없어 팀 최종 모델부터 비교합니다.")
                continue
            members[name] = member
        if path.is_file():
            probabilities[name] = aligned_probs(rows, path)
            print(f"  {name}: 저장 OOF 사용", flush=True)
        else:
            missing[name] = members[name]
    if missing:
        print(f"누락된 확률만 저장 가중치로 복원: {', '.join(missing)} (재학습 없음)", flush=True)
        probabilities.update(replay_missing(rows, frames, missing, args))

    variants = {"팀 최종": FINAL}
    if "rn18s" in probabilities:
        variants.update({"팀 rn18s 단독 · raw": {"rn18s": 1.0}, "팀 최종 + rn18s 10%": HYBRID})
    print(f"Train {len(frames['train']):,}행 / {frames['train'][KEY].nunique()}개 실험 · "
          f"Validation {len(frames['validation']):,}행 / {frames['validation'][KEY].nunique()}개 실험")
    result = []
    for name, weights in variants.items():
        p = sum(probabilities[m] * weight for m, weight in weights.items())
        labels = CLASSES[p.argmax(axis=1)]
        if weights != {"rn18s": 1.0}:
            labels = cummax_pp(rows, labels)
        result.append(metrics(rows, labels, name))
    print(pd.DataFrame(result).to_string(index=False, float_format="{:.5f}".format))
    base = result[0]
    if abs(base["OOF 5초"] - 0.9031) > 0.00005 or abs(base["전체 Val 5초"] - 0.9134) > 0.00005:
        print("팀 README 재현 기준 OOF 0.9031 / Val 0.9134와 차이가 있습니다. 사용한 OOF와 가중치 경로를 확인하세요.")
    else:
        print("팀 README의 5초 점수와 일치")
    if len(result) == 3:
        hybrid = result[-1]
        print(f"확장안 변화 · OOF {hybrid['OOF 5초'] - base['OOF 5초']:+.5f}"
              f" · Val {hybrid['전체 Val 5초'] - base['전체 Val 5초']:+.5f}"
              f" · 최저 fold {hybrid['최저 fold'] - base['최저 fold']:+.5f}"
              " · Public 미측정")
    print("rn18s는 이 폴드에서 다시 학습된 팀 멤버이고, Public 0.93057은 별도 ResNet18 v2 제출의 점수입니다.")


def submit(args, bundle):
    mode = args.make_submission
    if mode == "hybrid":
        rn18 = find_rn18(args)
        if rn18 is None:
            raise FileNotFoundError("rn18s 가중치가 필요합니다. --rn18_model 경로를 지정하세요.")
        members = {**bundle["members"], "rn18s": rn18}
        bundle = {**bundle, "members": members, "blend": {m: np.full(6, w) for m, w in HYBRID.items()}}
        bundle["feats"] = sorted({c for member in members.values() for c in (member.get("feats") or [])})

    data = args.data_dir.resolve()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        model = args.model.resolve()
        model_out = Path(temp) / "model_team_rn18_10.joblib"
        if mode == "hybrid":
            joblib.dump(bundle, model_out, compress=3)
            model = model_out
        csv = Path(temp) / f"submission_team_{mode}.csv"
        sample = data / "sample_submission.csv"
        flags = ["--input", str(data / "test/test.csv"), "--image_root", str(data / "test"), "--model", str(model)]
        subprocess.run([sys.executable, "-u", "test.py", *flags, "--sample_submission", str(sample),
                        "--output", str(csv), "--skip_check"], cwd=ROOT, check=True)
        result = pd.read_csv(csv, dtype={ID: str})
        expected = pd.read_csv(sample, dtype={ID: str})
        if (list(result.columns) != [ID, "label"] or not result[ID].is_unique
                or not pd.api.types.is_integer_dtype(result["label"])
                or not result[ID].equals(expected[ID]) or not result["label"].isin(CLASSES).all()):
            raise ValueError("생성 CSV가 제출 양식과 다릅니다.")
        if args.submitted_csv:
            prior = pd.read_csv(args.submitted_csv, dtype={ID: str}).set_index(ID)
            if not prior.index.is_unique or set(prior.index) != set(result[ID]):
                raise ValueError("기존 제출 CSV의 sample_id와 새 CSV의 sample_id가 다릅니다.")
            different = (prior.loc[result[ID], "label"].to_numpy() != result["label"].to_numpy()).sum()
            print(f"기존 제출 CSV와 {different:,}/{len(result):,}행 다름", flush=True)
            if different:
                raise ValueError("기존 제출 파일을 재현하지 못했습니다.")
        subprocess.run([sys.executable, "-u", "tools/check_submission.py", *flags, "--n_exp", "0"],
                       cwd=ROOT, check=True)
        if mode == "hybrid":
            model_out.replace(output_dir / model_out.name)
        csv.replace(output_dir / csv.name)
    print(f"제출 전 점검 통과 · {output_dir / csv.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path.home() / "data/battery_data_light")
    parser.add_argument("--stack_dir", type=Path, default=Path.home() / "unid-data-competition/runs/stack_trv",
                        help="팀 서버의 oof_lgb/dino/seq_gru/rn18s.csv 폴더")
    parser.add_argument("--model", type=Path, default=ROOT / "model.joblib")
    parser.add_argument("--rn18_model", type=Path)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "runs/compare_team_final")
    parser.add_argument("--make_submission", choices=("final", "hybrid"), help="Test CSV 생성 및 네 가지 점검")
    parser.add_argument("--submitted_csv", type=Path, help="실제 DAKER 제출 CSV와 대조 (선택)")
    args = parser.parse_args()
    if args.submitted_csv and not args.make_submission:
        parser.error("--submitted_csv는 --make_submission과 함께 씁니다.")
    bundle = load_final(args.model)
    if args.make_submission:
        submit(args, bundle)
    else:
        compare(args, bundle)


if __name__ == "__main__":
    main()
