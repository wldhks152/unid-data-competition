"""Build submission #34's saved model from the committed #27 baseline (no retraining)."""
import argparse
import hashlib
from pathlib import Path
import joblib

BASE_GIT_BLOB = "89bff3ac7fe9bb16338e11f7f8d6f756cbe9aa7a"
FINAL_SHA256 = "9dea0680d9abf08443aa6aab206dd90fff4e8f80125561ab99d40139c177f3d9"


def git_blob_sha(path):
    path = Path(path)
    sha = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("runs/team27_late/model_team27_late.joblib"))
    args = parser.parse_args()
    if not args.base.is_file() or git_blob_sha(args.base) != BASE_GIT_BLOB:
        raise ValueError("The base file is missing/truncated or is not the #27 baseline. Use commit cbd824a.")
    if args.out.resolve() == args.base.resolve():
        raise ValueError("Do not overwrite base: choose a separate --out destination.")
    bundle = joblib.load(args.base)
    if set(bundle.get("members", {})) != {"lgb", "dino", "seq_gru"}:
        raise ValueError("Unexpected base ensemble members")
    bundle = {**bundle, "post": {**bundle["post"], "method": "late", "module": "team27_late"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    actual = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"Exported {args.out} ({args.out.stat().st_size} bytes), SHA-256 {actual}")
    if actual != FINAL_SHA256:
        print("INFO: Serialization differs from the uploaded file; check the original Python/joblib versions and verify predictions against the submitted CSV.")


if __name__ == "__main__":
    main()
