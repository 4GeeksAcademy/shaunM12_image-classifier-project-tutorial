# Project summary

_Auto-generated quick view. See `reports/steps/` for detailed per-step reports._

> **Pipeline complete** — Steps 1–5 finished. Optional: Kaggle predict CSV if unlabeled test images are downloaded.

## Pipeline status

| Stage | Status | Time (UTC) | Duration (s) |
|-------|--------|------------|----------------|
| manifest | SUCCESS | 2026-07-23T14:31:26Z | 0 |
| split | SUCCESS | 2026-07-23T14:31:26Z | 0 |
| eda | SUCCESS | 2026-07-23T14:00:12Z | 7.97 |
| train (baseline) | SUCCESS | 2026-07-23T17:28:08Z | 8566 |
| optimize | SUCCESS | 2026-07-23T17:52:22Z | 8380 |
| save (Step 5) | SUCCESS | 2026-07-23T18:33:16Z | 1463.79 |
| leakage tests | 8/8 PASS | 2026-07-23T23:37:06Z | — |

## Data counts

| Split | Cats | Dogs | Total |
|-------|------|------|-------|
| train | 8747 | 8748 | 17495 |
| val | 1875 | 1874 | 3749 |
| test | 1875 | 1875 | 3750 |

- Raw images on disk: **25000**
- Kaggle unlabeled test images: **0**

## Model metrics

| Metric | Value |
|--------|-------|
| Baseline val accuracy (Step 3) | 98.29% |
| Best optimize val accuracy (Step 4) | 97.71% |
| Optimize best epoch | 1 |
| EarlyStopping triggered | False |
| Manual termination (Step 4) | True |
| Test accuracy (one-time) | 97.71% |
| Best checkpoint | `saved_models/best_model.keras` |
| Final export (Step 5) | `saved_models/asirra_cats_dogs_final.keras` |

## Key decisions

- Split seed: **42**
- Ratios: **train 70% / val 15% / test 15%**
- Image size: **224×224**
- Loss: **categorical_crossentropy** with 2-unit softmax
- Callbacks monitor: **val_accuracy** only
- Architecture note: The assignment refers to an ANN in the general sense; for image classification we implemented a convolutional neural network (VGG-style / VGG16 transfer), which is the appropriate architecture for photo-based cat vs dog classification.

## Leakage checks

- ✅ `test_content_hash_not_shared_across_splits` — Duplicate files stay in a single split
- ✅ `test_evaluate_gate_blocks_test_by_default` — Test evaluation is blocked unless ALLOW_TEST_EVAL=true
- ✅ `test_kaggle_test_dir_not_in_manifest` — Unlabeled Kaggle test images are excluded from manifest
- ✅ `test_manifest_exists_after_split` — Manifest exists with valid split column
- ✅ `test_processed_paths_under_processed_dir` — Train/val/test paths live under data/processed/
- ✅ `test_split_disjointness` — No image_id appears in more than one split
- ✅ `test_splits_json_matches_manifest` — splits.json counts match manifest.csv
- ✅ `test_stratification_within_tolerance` — Cat/dog ratio is similar across train, val, and test

## Visual aids (quick reference)

| Chart | Path |
|-------|------|
| Split counts (stratified) | `reports/eda/split_counts.png` |
| Class balance (train) | `reports/eda/class_balance_train.png` |
| Stratification ratios | `reports/eda/stratification_ratios.png` |
| Baseline training curves | `reports/metrics/training_curves.png` |
| Optimize training curves | `reports/metrics/optimized_training_curves.png` |
| EarlyStopping summary | `reports/metrics/early_stopping_summary.png` |
| Validation confusion matrix | `reports/metrics/val_confusion_matrix.png` |
| Validation ROC curve | `reports/metrics/val_roc_curve.png` |

### Key charts

**Split counts (stratified)** — [`eda/split_counts.png`](eda/split_counts.png)

![Split counts (stratified)](eda/split_counts.png)

**Baseline training curves** — [`metrics/training_curves.png`](metrics/training_curves.png)

![Baseline training curves](metrics/training_curves.png)

**EarlyStopping summary** — [`metrics/early_stopping_summary.png`](metrics/early_stopping_summary.png)

![EarlyStopping summary](metrics/early_stopping_summary.png)

**Validation confusion matrix** — [`metrics/val_confusion_matrix.png`](metrics/val_confusion_matrix.png)

![Validation confusion matrix](metrics/val_confusion_matrix.png)


## Warnings

- ⚠️ Images found in nested data/raw/asirra/train/train/ — split still works; flatten when convenient.

## Next step

```bash
# Pipeline complete — optional: download Kaggle test images + predict CSV
```
