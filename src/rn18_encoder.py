"""Frozen, offline ResNet18 image features with a verified three-part checkpoint.

The network layout follows torchvision's BSD-licensed ResNet18 implementation.
See third_party/LICENSE.torchvision. Torch is imported only when preparing or
running the encoder; manifest verification and image preprocessing need no torch.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
WEIGHTS_NAME = 'ResNet18_Weights.IMAGENET1K_V1'
CHECKPOINT_NAME = 'resnet18-f37072fd.pth'
HASH_PREFIX = 'f37072fd'
MAX_PART_BYTES = 16 * 1024 * 1024
EMBEDDING_COLUMNS = [f'resnet18_{i:03d}' for i in range(512)]
TRANSFORM_SPEC = {
    'name': 'rgb_letterbox224_imagenet_v1',
    'color_mode': 'RGB',
    'output_size': [224, 224],
    'resize': {
        'preserve_aspect_ratio': True,
        'long_edge': 224,
        'interpolation': 'PIL_BILINEAR',
        'rounding': 'floor(value + 0.5), minimum 1 pixel',
    },
    'padding': {
        'placement': 'center; extra odd pixel on right/bottom',
        'fill_rgb': [124, 116, 104],
    },
    'scale': 'float32 / 255',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'output_layout': 'CHW float32',
    'random_augmentation': False,
}


def _torch():
    import torch
    return torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _part_name(index: int) -> str:
    return f'{CHECKPOINT_NAME}.part{index:03d}'


def verify_encoder(manifest_path: Path, expected_sha256: str | None = None) -> dict:
    """Verify metadata, all part hashes, and the reassembled hash, without network.

    At inference pass the full hash saved independently in the trained model
    bundle. Trusting a hash solely from a replaceable manifest is insufficient.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Encoder manifest not found: {manifest_path}')
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError('Encoder manifest must contain a JSON object.')
    required = {
        'schema_version': 1,
        'architecture': 'resnet18',
        'weights_name': WEIGHTS_NAME,
        'source_url': SOURCE_URL,
        'embedding_dim': 512,
        'part_size_limit': MAX_PART_BYTES,
        'preprocess': TRANSFORM_SPEC,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f'Unsupported encoder manifest field: {key}')
    full_digest = metadata.get('full_sha256')
    if not _valid_digest(full_digest) or not full_digest.startswith(HASH_PREFIX):
        raise ValueError('Invalid official checkpoint SHA-256 in encoder manifest.')
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).lower()
        if not _valid_digest(expected_sha256):
            raise ValueError('Expected encoder SHA-256 must be 64 hexadecimal characters.')
        if full_digest != expected_sha256:
            raise ValueError('Encoder checkpoint differs from the trained model bundle.')

    parts = metadata.get('parts')
    if not isinstance(parts, list) or len(parts) != 3:
        raise ValueError('The official ResNet18 checkpoint must have three parts.')
    checkpoint_size = metadata.get('checkpoint_size')
    if type(checkpoint_size) is not int or checkpoint_size <= 0:
        raise ValueError('Invalid checkpoint_size in encoder manifest.')
    full_hasher = hashlib.sha256()
    total_size = 0
    parent = manifest_path.parent.resolve()
    for index, part in enumerate(parts, start=1):
        if not isinstance(part, dict) or part.get('name') != _part_name(index):
            raise ValueError('Invalid checkpoint part name or order.')
        size = part.get('size')
        if type(size) is not int or not 0 < size <= MAX_PART_BYTES:
            raise ValueError('Checkpoint part exceeds the 16 MiB limit or has invalid size.')
        if index < len(parts) and size != MAX_PART_BYTES:
            raise ValueError('Only the final checkpoint part may be shorter than 16 MiB.')
        if not _valid_digest(part.get('sha256')):
            raise ValueError('Invalid checkpoint part SHA-256.')
        path = manifest_path.parent / part['name']
        if path.resolve().parent != parent or not path.is_file():
            raise FileNotFoundError(f'Checkpoint part missing or outside encoder folder: {path.name}')
        if path.stat().st_size != size:
            raise ValueError(f'Checkpoint part size mismatch: {path.name}')
        part_hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                part_hasher.update(chunk)
                full_hasher.update(chunk)
                total_size += len(chunk)
        if part_hasher.hexdigest() != part['sha256']:
            raise ValueError(f'Checkpoint part checksum mismatch: {path.name}')
    if total_size != checkpoint_size or full_hasher.hexdigest() != full_digest:
        raise ValueError('Reassembled encoder checkpoint checksum or size mismatch.')
    return metadata


def prepare_encoder(weights_dir: Path) -> dict:
    """Download once, verify, and split official weights for GitHub web uploads.

    This is the only function in this module that can access the network.
    The full downloaded checkpoint lives in a temporary directory outside the
    package. Only three <=16 MiB parts and their manifest remain in weights/.
    An existing manifest is verified and reused; damaged weights are not silently
    replaced. Run this preparation on the competition-provided server.
    """
    encoder_dir = Path(weights_dir) / 'encoder'
    manifest_path = encoder_dir / 'manifest.json'
    if manifest_path.exists():
        return verify_encoder(manifest_path)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    with tempfile.TemporaryDirectory(prefix='uni_d_resnet18_') as scratch:
        checkpoint = Path(scratch) / CHECKPOINT_NAME
        print(f'Downloading official encoder weights: {SOURCE_URL}', flush=True)
        torch.hub.download_url_to_file(
            SOURCE_URL, str(checkpoint), hash_prefix=HASH_PREFIX, progress=True,
        )
        full_digest = _sha256_file(checkpoint)
        if not full_digest.startswith(HASH_PREFIX):
            raise ValueError('Downloaded encoder checkpoint has an invalid SHA-256 prefix.')
        checkpoint_size = checkpoint.stat().st_size
        n_parts = (checkpoint_size + MAX_PART_BYTES - 1) // MAX_PART_BYTES
        if n_parts != 3:
            raise ValueError('Unexpected official checkpoint size; expected three <=16 MiB parts.')
        parts = []
        with checkpoint.open('rb') as stream:
            for index in range(1, n_parts + 1):
                chunk = stream.read(MAX_PART_BYTES)
                name = _part_name(index)
                staged = Path(scratch) / name
                staged.write_bytes(chunk)
                target_tmp = encoder_dir / f'{name}.tmp'
                shutil.copyfile(staged, target_tmp)
                target_tmp.replace(encoder_dir / name)
                parts.append({
                    'name': name,
                    'sha256': hashlib.sha256(chunk).hexdigest(),
                    'size': len(chunk),
                })
        metadata = {
            'schema_version': 1,
            'architecture': 'resnet18',
            'weights_name': WEIGHTS_NAME,
            'source_url': SOURCE_URL,
            'full_sha256': full_digest,
            'checkpoint_size': checkpoint_size,
            'embedding_dim': 512,
            'part_size_limit': MAX_PART_BYTES,
            'preprocess': json.loads(json.dumps(TRANSFORM_SPEC)),
            'parts': parts,
        }
        manifest_tmp = encoder_dir / 'manifest.json.tmp'
        manifest_tmp.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        manifest_tmp.replace(manifest_path)
    return verify_encoder(manifest_path)


def resolve_device(requested: str) -> str:
    if requested not in ('auto', 'cuda', 'cpu'):
        raise ValueError('Encoder device must be auto, cuda, or cpu.')
    if requested == 'cpu':
        return 'cpu'
    torch = _torch()
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise RuntimeError('CUDA was requested but is unavailable in the provided environment.')
    return 'cuda' if available else 'cpu'


def _configure_determinism(torch):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')


def runtime_info(device: str) -> dict:
    torch = _torch()
    return {
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'cpu_threads': torch.get_num_threads(),
        'embedding_batch_size': 1,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }


def preprocess_image(path: Path) -> np.ndarray:
    """Return deterministic RGB-letterboxed, normalized CHW float32 pixels."""
    with Image.open(path) as source:
        image = source.convert('RGB')
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image dimensions: {path}')
    scale = 224.0 / max(width, height)
    resized_width = min(224, max(1, int(math.floor(width * scale + 0.5))))
    resized_height = min(224, max(1, int(math.floor(height * scale + 0.5))))
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (224, 224), (124, 116, 104))
    canvas.paste(image, ((224 - resized_width) // 2, (224 - resized_height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    pixels -= np.asarray(TRANSFORM_SPEC['mean'], dtype=np.float32)
    pixels /= np.asarray(TRANSFORM_SPEC['std'], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)


def _build_resnet18(torch):
    """Construct torchvision-compatible state keys, including the original fc."""
    nn = torch.nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 2)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes
            layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))

    return ResNet18()


class FrozenResNet18:
    def __init__(self, manifest_path: Path, device: str, expected_sha256: str | None = None):
        self.manifest_path = Path(manifest_path)
        self.metadata = verify_encoder(self.manifest_path, expected_sha256=expected_sha256)
        self.device = resolve_device(device)
        torch = _torch()
        _configure_determinism(torch)
        # Reassemble only in memory; inference never downloads or writes weights.
        checkpoint = io.BytesIO()
        full_hasher = hashlib.sha256()
        for part in self.metadata['parts']:
            with (self.manifest_path.parent / part['name']).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    checkpoint.write(chunk)
                    full_hasher.update(chunk)
        if full_hasher.hexdigest() != self.metadata['full_sha256']:
            raise ValueError('Encoder checkpoint changed during loading.')
        checkpoint.seek(0)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = _build_resnet18(torch)
        # Validate the complete official checkpoint, including fc1000, first.
        self.model.load_state_dict(state, strict=True)
        self.model.fc = torch.nn.Identity()
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device=self.device, dtype=torch.float32)

    def extract(self, paths, progress_label: str = 'images') -> np.ndarray:
        """Extract one current image at a time, preserving the supplied order."""
        paths = list(paths)
        result = np.empty((len(paths), 512), dtype=np.float32)
        torch = _torch()
        self.model.eval()
        with torch.inference_mode():
            for index, path in enumerate(paths):
                pixels = preprocess_image(Path(path))
                inputs = torch.from_numpy(pixels).unsqueeze(0).to(device=self.device, dtype=torch.float32)
                features = self.model(inputs)
                if tuple(features.shape) != (1, 512):
                    raise ValueError(f'Unexpected encoder output shape: {tuple(features.shape)}')
                values = features[0].to(device='cpu', dtype=torch.float32).numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f'Encoder returned non-finite features at input row {index}.')
                result[index] = values
                if (index + 1) % 500 == 0 or index + 1 == len(paths):
                    print(f'[{progress_label}] {index + 1:,}/{len(paths):,} images (batch=1)', flush=True)
        return result
