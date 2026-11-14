"""torchvision 공식 ResNet50 ImageNet-1K 가중치(IMAGENET1K_V2)를 한 번 내려받아 16MiB 조각으로 weights/resnet50/에 저장하고,
출처·버전·조각별/전체 SHA-256을 manifest.json에 남긴다. 학습 전에 한 번만 실행한다. 추론(test.py)은 이 폴더만 읽는다.

    python tools/get_resnet50.py
"""
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.r50 import CHECKPOINT_NAME, HASH_PREFIX, PART_BYTES, R50_DIR, SOURCE_URL, WEIGHTS_NAME  # noqa: E402


def main():
    if (R50_DIR / "manifest.json").exists():
        from src.r50 import load_checkpoint_bytes
        load_checkpoint_bytes()
        print(f"이미 있음, 해시 확인 완료 → {R50_DIR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / CHECKPOINT_NAME
        print(f"내려받기 {SOURCE_URL}", flush=True)
        urllib.request.urlretrieve(SOURCE_URL, f)
        data = f.read_bytes()
    full = hashlib.sha256(data).hexdigest()
    if not full.startswith(HASH_PREFIX):
        raise SystemExit(f"SHA-256이 공식 파일명({HASH_PREFIX})과 다릅니다: {full[:12]}")
    R50_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, s in enumerate(range(0, len(data), PART_BYTES), start=1):
        b = data[s:s + PART_BYTES]
        name = f"{CHECKPOINT_NAME}.part{i:03d}"
        (R50_DIR / name).write_bytes(b)
        parts.append({"name": name, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)})
    meta = {"architecture": "resnet50", "weights_name": WEIGHTS_NAME, "source_url": SOURCE_URL,
            "license": "torchvision BSD-3-Clause (code); weights per torchvision distribution terms",
            "full_sha256": full, "checkpoint_size": len(data), "embedding_dim": 2048, "parts": parts,
            "use": "frozen feature extractor (global average pool, 2048-d), no fine-tuning"}
    (R50_DIR / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"저장 → {R50_DIR}  {len(data) / 1e6:.1f}MB, 조각 {len(parts)}개, sha256 {full[:12]}")


if __name__ == "__main__":
    main()
