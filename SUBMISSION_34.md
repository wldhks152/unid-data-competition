# Uni-DTHON submission #34 — `submission_team27_late.csv`

> #34 is the **submission number**, not model version 34. Public score from submitted screenshot: **0.9004177879** (2026-11-15 04:49 KST). This is not a validation-set estimate.

## Model recipe

- Starting source revision: `cbd824a0cadaa805f397f02e6ebf3d692862ed03` on `diag/public-drop`.
- Base model: LightGBM 0.4 + DINOv2-small/thermal-image CatBoost 0.4 + causal GRU 0.2.
- `team27_late.py` contains causal `late` postprocessing: for upward transitions into stages 4–6, require strong confidence (tail probability >= 0.75) or support from the immediately preceding observation within 15 s; then per-experiment cumulative maximum. Initial observation is kept.
- Saved final artifact: `runs/team27_late/model_team27_late.joblib` (the uploaded ZIP's working final model), loaded under `model.joblib` in the deployment directory.

## Evidence

- Submitted CSV: 3,451 rows, schema `sample_id,label`, integer labels 1–6.
- Submitted CSV SHA-256: `f0068bef638aaf1f5128e54e4bf289d8c13b959d8994277c0ce97ac84625c59c` (uploaded and archived copies match).
- Saved final model SHA-256: `9dea0680d9abf08443aa6aab206dd90fff4e8f80125561ab99d40139c177f3d9` (42,454,870 bytes).
- Prior GPU-server logs: Train OOF 5s Macro-F1 0.90250; held-out Validation 5s 0.91435. Replay of the 3,451 test rows and the four submission checks passed then. This GitHub commit does **not** independently rerun training or GPU inference.
- Original #27 baseline OOF 0.90310, Validation 0.91340, so the postprocessor traded a small OOF decline for a small Validation improvement. Public score for #34 comes from the screenshot, not from our local evaluation.

## Regenerate the #34 model without retraining

Use this branch's untouched baseline `model.joblib` (Git blob `89bff3ac7fe9bb16338e11f7f8d6f756cbe9aa7a`) and execute on the original Python 3.10 environment:

```bash
python tools/export_submission34.py --base model.joblib --out runs/team27_late/model_team27_late.joblib
```

The exporter performs a small metadata change: `post.method = "late"`, `post.module = "team27_late"`, preserving the trained ensemble weights. To submit using the standard evaluator CLI, **replace root `model.joblib`** with this exported artifact:

```bash
cp runs/team27_late/model_team27_late.joblib model.joblib
python test.py --input "$HOME/data/battery_data_light/test/test.csv" \
  --image_root "$HOME/data/battery_data_light/test" \
  --output runs/team27_late/submission_team27_late.csv \
  --sample_submission "$HOME/data/battery_data_light/sample_submission.csv"
python tools/check_submission.py --input "$HOME/data/battery_data_light/test/test.csv" \
  --image_root "$HOME/data/battery_data_light/test" --model model.joblib --n_exp 0
```

Do **not** use the `model.joblib` at the root of the user's uploaded ZIP: it has exactly 25MiB and is truncated. The `runs/team27_late/model_team27_late.joblib` inside that ZIP is complete. Do not upload competition training/validation/test CSV or thermal images to GitHub.

## Publication status

The branch is created as a non-disruptive fork of the verified baseline commit. GitHub's small-file connector can upload source/scripts and documentation, but the 42MB final model needs an explicit Git push from an environment with its bytes. Until that push happens, `model.joblib` on GitHub remains the **base** cummax model and is **not equivalent** to submission #34. Don't treat this branch as the complete final deliverable until `model.joblib` SHA-256 matches the final hash above.
