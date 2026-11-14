"""팀원 ResNet18 + CatBoost v1을 스태킹 멤버(rn18)로 — 같은 fold로 학습해 OOF를 만든다.

* 이미지 특징: ImageNet 사전학습 ResNet18 (torchvision ResNet18_Weights.IMAGENET1K_V1, 동결, avgpool 512차원).
  가중치 출처 https://download.pytorch.org/models/resnet18-f37072fd.pth — 대회 규정 4(대회 전 공개된 범용
  사전학습 모델 허용, README에 출처 기재)에 따라 사용. 구현·전처리는 src/rn18_encoder.py (팀원 코드, BSD-3 고지는
  third_party/LICENSE.torchvision), 가중치 조각은 weights/encoder/.
* 표 특징: 팀원 CatBoost v1의 33개 (src/team_catboost.build_features) + 512개 = 545개.
* 이미지 특징은 행(이미지)마다 독립 계산 → 미래 정보 없음, 캐시해도 안전.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import team_catboost as TM

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "weights" / "encoder" / "manifest.json"
EMB_COLS = [f"resnet18_{i:03d}" for i in range(512)]
CLASSES = np.arange(1, 7)
_ENC = {}
_MEMO = {}          # 프로세스 안 경로별 임베딩 메모 (같은 이미지 파일을 다시 읽지 않음)
IMAGE_ROOT = None   # test.py가 --image_root로 채운다


def _encoder(device="auto"):
    if device not in _ENC:
        from .rn18_encoder import FrozenResNet18
        _ENC[device] = FrozenResNet18(MANIFEST, device)
    return _ENC[device]


def embeddings(df, image_root, cache_dir=None, tag=None, device="auto"):
    """sample_id 인덱스의 (n, 512) DataFrame. cache_dir·tag가 있으면 npz로 캐시."""
    cache = os.path.join(cache_dir, f"rn18_{tag}.npz") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        E = pd.DataFrame(z["E"], index=pd.Index(z["ids"], name="sample_id"), columns=EMB_COLS)
        if set(df.sample_id) <= set(E.index):
            return E.loc[df.sample_id.values]
    root = Path(image_root)
    paths = [str(Path(p).resolve()) for p in TM.resolve_image_paths(df, [root, root.parent])]
    todo = sorted({p for p in paths if p not in _MEMO})
    if todo:
        for p, e in zip(todo, _encoder(device).extract(todo, progress_label=tag or "images")):
            _MEMO[p] = e
    E = np.stack([_MEMO[p] for p in paths]).astype(np.float32) if paths else np.zeros((0, 512), np.float32)
    out = pd.DataFrame(E, index=pd.Index(df.sample_id.values, name="sample_id"), columns=EMB_COLS)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, E=E, ids=np.asarray(df.sample_id.values, dtype=object))
    return out


def build_X(frame, timg, emb):
    base = TM.build_features(frame, timg)
    E = emb.reindex(frame["sample_id"].values)
    assert E.notna().all(axis=1).all(), "ResNet18 특징이 없는 샘플이 있습니다."
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(E.to_numpy(np.float32), columns=EMB_COLS)], axis=1)


def fit(raw, timg, emb, thread_count=None):
    X = build_X(raw, timg, emb)
    pipe = TM.build_model(thread_count=thread_count or os.cpu_count() or 4)
    pipe.fit(X, raw["label"].astype(int).to_numpy())
    return {"features": list(X.columns), "pipeline": pipe}


def proba(model, raw, timg, emb, sample_ids):
    frame = raw.set_index("sample_id").loc[list(sample_ids)].reset_index()
    X = build_X(frame, timg, emb)[model["features"]]
    P = model["pipeline"].predict_proba(X)
    tc = [int(c) for c in model["pipeline"].classes_]
    return P[:, [tc.index(int(c)) for c in CLASSES]]


def predict_proba(member, F, df, timg=None):
    """blend.py custom 멤버 추론. test.py가 IMAGE_ROOT(또는 df.attrs["image_root"])에 이미지 폴더를 넣어 준다."""
    root = df.attrs.get("image_root") or IMAGE_ROOT
    assert root, "rn18 멤버는 이미지 폴더가 필요합니다 (test.py --image_root)."
    if timg is None:
        from .ensemble import team_image_features
        timg = team_image_features(df, root)
    emb = embeddings(df, root, device=member.get("device", "auto"))
    return np.mean([proba(m, df, timg, emb, F.sample_id.values) for m in member["models"]], axis=0)
