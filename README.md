# Asirra Cats vs Dogs — Leak-Free CNN Pipeline

Keras transfer-learning project (VGG16) for the [Asirra / Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset. The pipeline enforces a **leak-free** train / validation / test protocol and writes **per-step reports** for review.

See [`specs.md`](specs.md) for the full data policy and tutorial requirements.

## Assignment alignment (ANN vs CNN)

The assignment refers to an ANN in the general sense; for image classification we implemented a **convolutional neural network** (VGG-style / VGG16 transfer), which is the appropriate architecture for photo-based cat vs dog classification. Step 3 of the tutorial specifies `Conv2D` / `MaxPool2D` layers (a CNN), not a dense-only network. Use `--from-scratch` to train the literal VGG-style Sequential stack from the tutorial.

## Quick status

After running the pipeline, open **[`reports/PROJECT_SUMMARY.md`](reports/PROJECT_SUMMARY.md)** for a one-page view: stage status, metrics, leakage tests, and key charts.

Detailed step narratives live in [`reports/steps/`](reports/steps/).

## Data layout

```
data/raw/asirra/train/     ← ~25k labeled images (immutable; do not train from here after split)
data/raw/asirra/test/      ← ~12.5k unlabeled Kaggle images (optional inference only)
data/interim/              ← manifest.csv, splits.json, dedup_report.csv
data/processed/train/      ← 70% stratified copy (generators train from here)
data/processed/val/        ← 15% — EarlyStopping, ModelCheckpoint, tuning
data/processed/test/       ← 15% — one-time final evaluation only
saved_models/              ← baseline, best checkpoint, final export (.keras)
reports/                   ← auto-generated step reports, charts, tests
submissions/               ← optional Kaggle prediction CSV
```

## Split & leakage rules

| Split | Share | Use |
|-------|-------|-----|
| Train | 70% | `model.fit()` only |
| Val | 15% | Callbacks, charts, reload sanity check |
| Test | 15% | **One** gated evaluation (`ALLOW_TEST_EVAL=true`) |

- Stratified by class, seed **42**, deduplicated by content hash.
- **Never** train from `data/raw/` after the split script runs.
- **Never** use Kaggle unlabeled `test/` for accuracy or callbacks.
- `ModelCheckpoint` and `EarlyStopping` monitor **`val_accuracy`** only.

## Pipeline commands

Run from the project root:

```bash
# Step 1 — manifest + stratified split
python src/app.py --stage manifest
python src/app.py --stage split

# Step 2 — EDA
python src/app.py --stage eda

# Step 3 — baseline VGG16 transfer learning (default 1 epoch on CPU)
python src/app.py --stage train

# Charts only (no retraining)
python src/app.py --stage visuals

# Step 4 — optimize with EarlyStopping + ModelCheckpoint
python src/app.py --stage optimize --epochs 6 --patience 2

# Step 4 — one-time held-out test (locked until explicit opt-in)
ALLOW_TEST_EVAL=true python src/app.py --stage evaluate

# Step 5 — export final model + val reload check
python src/app.py --stage save

# Refresh reports, charts, leakage tests, PROJECT_SUMMARY.md
python src/app.py --stage report
```

Shortcut for data prep only:

```bash
python src/app.py --stage all   # manifest + split
```

## Setup

**Codespaces (recommended):** environment configures automatically. Install deps if needed:

```bash
pip install -r requirements.txt
```

**Local:** Python 3.11+, then `pip install -r requirements.txt`. Download Asirra labeled train images into `data/raw/asirra/train/` (see [`data/raw/asirra/README.md`](data/raw/asirra/README.md)).

## Key outputs

| Artifact | Path |
|----------|------|
| Baseline model | `saved_models/baseline_model.keras` |
| Best checkpoint (Step 4) | `saved_models/best_model.keras` |
| Final export (Step 5) | `saved_models/asirra_cats_dogs_final.keras` |
| Test metrics (once) | `reports/metrics/final_test.json` |
| EarlyStopping chart | `reports/metrics/early_stopping_summary.png` |
| Leakage tests | `reports/tests/test_results.md` |

## What is gitignored

Large and generated artifacts stay local (see [`.gitignore`](.gitignore)):

- Raw and processed images
- Interim manifest/splits CSV/JSON
- Trained `.keras` models
- Generated reports, charts, and `PROJECT_SUMMARY.md`

Tracked in git: **source code**, `specs.md`, `submissions/sample_submission.csv`, and leakage **test result summaries** under `reports/tests/`.

## Optional: Kaggle submission

Download unlabeled Kaggle test images to `data/raw/asirra/test/`, then run predict (when implemented):

```bash
python src/app.py --stage predict
```

Output: `submissions/asirra_predictions.csv` (inference only — no accuracy on that folder).

## Tests

```bash
python -m unittest tests.test_leakage -v
```

## Project structure

```
src/
  app.py              # CLI entry point
  config.py           # paths, hyperparameters, leakage test descriptions
  data/               # manifest, split, EDA, generators
  models/             # VGG build, train, evaluate, save
  visualizations.py   # EDA + training + validation charts
  reporting.py        # step reports + PROJECT_SUMMARY.md
tests/
  test_leakage.py     # split / path / evaluate-gate checks
```
