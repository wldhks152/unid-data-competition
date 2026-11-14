"""DINOv2-small(동결) 이미지 특징 — rn18 멤버의 ResNet18 임베딩 대신 쓰는 384차원 CLS(pooler) 특징.

* 가중치: Hugging Face `facebook/dinov2-small` (Meta AI, Apache-2.0, 대회 전 공개 범용 사전학습 모델).
  tools/get_dinov2.py로 weights/dinov2-small/에 한 번 내려받고, 추론 때는 내려받지 않는다 (local_files_only).
* 전처리: RGB → 224×224 bilinear → ImageNet 평균·표준편차 정규화. 무작위 증강 없음.
* 이미지마다 독립 계산이라 캐시해도 안전하다. 배치는 항상 같은 크기로 채워 돌려서 행 순서와 무관하게 같은 값이 나온다.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import team_catboost as TM

ROOT = Path(__file__).resolve().parent.parent
DINO_DIR = ROOT / "weights" / "dinov2-small"
REPO_ID = "facebook/dinov2-small"
DIM = 384
EMB_COLS = [f"dino_{i:03d}" for i in range(DIM)]
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
BATCH = 128
_MODEL = {}
_MEMO = {}


def _device(requested="auto"):
    import torch
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model(device="auto"):
    if device not in _MODEL:
        import torch
        from transformers import Dinov2Model
        from .rn18_encoder import _configure_determinism
        _configure_determinism(torch)
        if not (DINO_DIR / "model.safetensors").exists():
            raise FileNotFoundError(f"DINOv2 가중치가 없습니다: {DINO_DIR} (먼저 tools/get_dinov2.py 실행)")
        m = Dinov2Model.from_pretrained(str(DINO_DIR), local_files_only=True, attn_implementation="eager")
        m.requires_grad_(False)
        m.eval()
        dev = _device(device)
        _MODEL[device] = (m.to(dev, dtype=torch.float32), dev)
    return _MODEL[device]


def preprocess(path):
    with Image.open(path) as im:
        x = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
    return (x.transpose(2, 0, 1) - MEAN) / STD


def _extract(paths, label, device="auto", batch=BATCH):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    model, dev = _model(device)
    out = np.empty((len(paths), DIM), np.float32)
    with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex, torch.inference_mode():
        for s in range(0, len(paths), batch):
            chunk = paths[s:s + batch]
            x = np.zeros((batch, 3, SIZE, SIZE), np.float32)
            x[:len(chunk)] = np.stack(list(ex.map(preprocess, chunk)))
            f = model(pixel_values=torch.from_numpy(x).to(dev)).pooler_output.float().cpu().numpy()[:len(chunk)]
            if not np.isfinite(f).all():
                raise ValueError("DINOv2 특징에 NaN/inf")
            out[s:s + len(chunk)] = f
            done = s + len(chunk)
            if done % (batch * 40) < batch or done == len(paths):
                print(f"[{label}] DINOv2 {done:,}/{len(paths):,} images (batch={batch})", flush=True)
    return out


def embeddings(df, image_root, cache_dir=None, tag=None, device="auto"):
    """sample_id 인덱스의 (n, 384) DataFrame. cache_dir·tag가 있으면 npz로 캐시."""
    cache = os.path.join(cache_dir, f"dino_{tag}.npz") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        E = pd.DataFrame(z["E"], index=pd.Index(z["ids"], name="sample_id"), columns=EMB_COLS)
        if set(df.sample_id) <= set(E.index):
            return E.loc[df.sample_id.values]
    root = Path(image_root)
    paths = [str(Path(p).resolve()) for p in TM.resolve_image_paths(df, [root, root.parent])]
    todo = sorted({p for p in paths if p not in _MEMO})
    if todo:
        for p, e in zip(todo, _extract(todo, tag or "images", device)):
            _MEMO[p] = e
    E = np.stack([_MEMO[p] for p in paths]).astype(np.float32) if paths else np.zeros((0, DIM), np.float32)
    out = pd.DataFrame(E, index=pd.Index(df.sample_id.values, name="sample_id"), columns=EMB_COLS)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, E=E, ids=np.asarray(df.sample_id.values, dtype=object))
    return out


def manifest():
    p = DINO_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
