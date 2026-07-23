# Leakage test results

- **Timestamp (UTC):** 2026-07-23T23:37:06Z
- **Command:** `python -m unittest tests.test_leakage -v`
- **Totals:** 8 passed, 0 failed, 0 skipped

| Test | Status | What it verifies | Why it matters |
|------|--------|------------------|----------------|
| `test_content_hash_not_shared_across_splits` | ✅ PASS | Duplicate files stay in a single split | Prevents duplicate-image leakage across splits |
| `test_evaluate_gate_blocks_test_by_default` | ✅ PASS | Test evaluation is blocked unless ALLOW_TEST_EVAL=true | Prevents accidental repeated test peeking |
| `test_kaggle_test_dir_not_in_manifest` | ✅ PASS | Unlabeled Kaggle test images are excluded from manifest | Keeps unlabeled data out of labeled split logic |
| `test_manifest_exists_after_split` | ✅ PASS | Manifest exists with valid split column | Ensures pipeline metadata is complete before training |
| `test_processed_paths_under_processed_dir` | ✅ PASS | Train/val/test paths live under data/processed/ | Prevents accidental training from raw data paths |
| `test_split_disjointness` | ✅ PASS | No image_id appears in more than one split | Prevents the same image from being in train and test |
| `test_splits_json_matches_manifest` | ✅ PASS | splits.json counts match manifest.csv | Ensures reporting and materialized splits are consistent |
| `test_stratification_within_tolerance` | ✅ PASS | Cat/dog ratio is similar across train, val, and test | Prevents biased splits that distort evaluation |

**Summary:** 8 passed, 0 failed, 0 skipped.
