"""DINOv2-small 공개 가중치를 weights/dinov2-small/에 한 번 내려받고 출처·버전·해시를 manifest.json에 남긴다.
학습 전에 한 번만 실행한다. 추론(test.py)은 이 폴더만 읽고 내려받지 않는다.

    python tools/get_dinov2.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dino import DINO_DIR, REPO_ID  # noqa: E402


def main():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO_ID)
    DINO_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO_ID, revision=info.sha, local_dir=str(DINO_DIR),
                      allow_patterns=["config.json", "model.safetensors", "README.md"])
    f = DINO_DIR / "model.safetensors"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    meta = {"repo_id": REPO_ID, "revision": info.sha, "source_url": f"https://huggingface.co/{REPO_ID}",
            "license": "apache-2.0", "file": f.name, "size": f.stat().st_size, "sha256": sha,
            "use": "frozen feature extractor (pooler_output, 384-d), no fine-tuning"}
    (DINO_DIR / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"저장 → {DINO_DIR}  revision {info.sha[:12]}  {f.stat().st_size / 1e6:.1f}MB  sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
