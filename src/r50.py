"""ResNet50(동결, ImageNet-1K 사전학습) 이미지 특징 — dino 멤버의 DINOv2 대신 쓰는 2048차원 평균풀링 특징.

* 가중치: torchvision 공식 배포 `ResNet50_Weights.IMAGENET1K_V2` (resnet50-11ad3fa6.pth, BSD-3-Clause 코드,
  대회 전 공개 범용 사전학습 모델). tools/get_resnet50.py로 한 번 내려받아 16MiB 조각으로 weights/resnet50/에 두고,
  추론 때는 조각을 이어 붙여 SHA-256을 확인한 뒤 읽는다 (내려받지 않음).
* 망 구조: torchvision ResNet50(v1.5, 3×3 conv에 stride)과 같은 층 이름으로 여기서 정의 → torchvision 없이도 state_dict를 읽는다.
  fc 층은 쓰지 않는다.
* 전처리: RGB → 224×224 bilinear → ImageNet 평균·표준편차 정규화. 무작위 증강 없음. dino.py와 같다.
* 이미지마다 독립 계산이라 캐시해도 안전하다. 배치는 항상 같은 크기로 채워 돌려 행 순서와 무관하게 같은 값이 나온다.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import team_catboost as TM
from .dino import preprocess  # 같은 전처리 (224 bilinear, ImageNet 정규화)

ROOT = Path(__file__).resolve().parent.parent
R50_DIR = ROOT / "weights" / "resnet50"
SOURCE_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
WEIGHTS_NAME = "ResNet50_Weights.IMAGENET1K_V2"
CHECKPOINT_NAME = "resnet50-11ad3fa6.pth"
HASH_PREFIX = "11ad3fa6"
PART_BYTES = 16 * 1024 * 1024
DIM = 2048
EMB_COLS = [f"r50_{i:04d}" for i in range(DIM)]
SIZE = 224
BATCH = 128
_MODEL = {}
_MEMO = {}


def _net():
    import torch.nn as nn

    class Bottleneck(nn.Module):
        def __init__(self, inp, width, stride, down):
            super().__init__()
            self.conv1 = nn.Conv2d(inp, width, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)
            self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(width)
            self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(width * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = (nn.Sequential(nn.Conv2d(inp, width * 4, 1, stride, bias=False), nn.BatchNorm2d(width * 4))
                               if down else None)

        def forward(self, x):
            idt = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            return self.relu(self.bn3(self.conv3(out)) + idt)

    class ResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            inp = 64
            for i, (width, n, stride) in enumerate([(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)], start=1):
                blocks = []
                for j in range(n):
                    blocks.append(Bottleneck(inp, width, stride if j == 0 else 1, j == 0))
                    inp = width * 4
                setattr(self, f"layer{i}", nn.Sequential(*blocks))

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return x.mean(dim=(2, 3))  # = AdaptiveAvgPool2d(1) + flatten, 2048차원

    return ResNet50()


def manifest():
    p = R50_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_checkpoint_bytes():
    """조각을 이어 붙이고 조각별·전체 SHA-256을 확인한다 (네트워크 없음)."""
    meta = manifest()
    if meta is None:
        raise FileNotFoundError(f"ResNet50 가중치가 없습니다: {R50_DIR} (먼저 tools/get_resnet50.py 실행)")
    if meta.get("source_url") != SOURCE_URL or not str(meta.get("full_sha256", "")).startswith(HASH_PREFIX):
        raise ValueError("ResNet50 manifest가 공식 체크포인트와 다릅니다")
    buf = bytearray()
    for part in meta["parts"]:
        b = (R50_DIR / part["name"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != part["sha256"]:
            raise ValueError(f"조각 해시 불일치: {part['name']}")
        buf += b
    if len(buf) != meta["checkpoint_size"] or hashlib.sha256(buf).hexdigest() != meta["full_sha256"]:
        raise ValueError("이어 붙인 ResNet50 체크포인트 해시 불일치")
    return bytes(buf)


def _model(device="auto"):
    if device not in _MODEL:
        import torch
        from .rn18_encoder import _configure_determinism
        _configure_determinism(torch)
        sd = torch.load(io.BytesIO(load_checkpoint_bytes()), map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        m = _net()
        m.load_state_dict(sd, strict=True)
        m.requires_grad_(False)
        m.eval()
        dev = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        _MODEL[device] = (m.to(dev, dtype=torch.float32), dev)
    return _MODEL[device]


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
            f = model(torch.from_numpy(x).to(dev)).float().cpu().numpy()[:len(chunk)]
            if not np.isfinite(f).all():
                raise ValueError("ResNet50 특징에 NaN/inf")
            out[s:s + len(chunk)] = f
            done = s + len(chunk)
            if done % (batch * 40) < batch or done == len(paths):
                print(f"[{label}] ResNet50 {done:,}/{len(paths):,} images (batch={batch}, {dev})", flush=True)
    return out


def embeddings(df, image_root, cache_dir=None, tag=None, device="auto"):
    """sample_id 인덱스의 (n, 2048) DataFrame. cache_dir·tag가 있으면 npz로 캐시."""
    cache = os.path.join(cache_dir, f"r50_{tag}.npz") if cache_dir and tag else None
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
