# Reports

Auto-generated documentation for the Asirra cats vs dogs pipeline. See **specs.md §2** for the full reporting spec.

The assignment rubric says “ANN” in the general neural-network sense; Step 3 implements a **CNN** (VGG-style / VGG16 transfer). See [`step03_train_baseline.md`](steps/step03_train_baseline.md) and [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) for the full note.

## Quick view

| File | Purpose |
|------|---------|
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | One-page status: pipeline stages, metrics, charts, test pass/fail, next step |
| [`tests/test_results.md`](tests/test_results.md) | All leakage tests with PASS/FAIL and reasoning |
| [`steps/`](steps/) | Detailed per-step `.md` + `.json` reports |

## Per-step reports

| File | Stage |
|------|-------|
| `steps/step01_manifest.md` | Scan raw images, build manifest, dedup |
| `steps/step015_split.md` | Stratified split + copy to `data/processed/` |
| `steps/step02_eda.md` | EDA grids, class balance, generators |
| `steps/step03_train_baseline.md` | VGG16 baseline training + validation metrics |
| `steps/step04_optimize.md` | EarlyStopping, ModelCheckpoint, test eval summary |
| `steps/step05_save.md` | Final model export + reload sanity check |

## Chart folders

| Path | Created by |
|------|------------|
| `eda/` | Step 2 EDA — split counts, class balance, dimensions, augmentation preview |
| `metrics/` | Steps 3–5 — training curves, EarlyStopping summary, validation error analysis |

## Regenerate reports (no retraining)

```bash
python src/app.py --stage visuals   # refresh charts from disk
python src/app.py --stage report    # leakage tests + PROJECT_SUMMARY.md + step04 chart refresh
```

Full pipeline commands are in the root [`README.md`](../README.md).
