"""Project paths and hyperparameters — see specs.md for data policy."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data (immutable downloads)
RAW_TRAIN_DIR = PROJECT_ROOT / "data/raw/asirra/train"
RAW_KAGGLE_TEST_DIR = PROJECT_ROOT / "data/raw/asirra/test"

# Processed splits (generators read these only)
PROCESSED_TRAIN_DIR = PROJECT_ROOT / "data/processed/train"
PROCESSED_VAL_DIR = PROJECT_ROOT / "data/processed/val"
PROCESSED_TEST_DIR = PROJECT_ROOT / "data/processed/test"

# Interim metadata
INTERIM_DIR = PROJECT_ROOT / "data/interim"
MANIFEST_PATH = INTERIM_DIR / "manifest.csv"
SPLITS_PATH = INTERIM_DIR / "splits.json"
DEDUP_REPORT_PATH = INTERIM_DIR / "dedup_report.csv"

# Submissions
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SAMPLE_SUBMISSION_PATH = SUBMISSIONS_DIR / "sample_submission.csv"
PREDICTIONS_PATH = SUBMISSIONS_DIR / "asirra_predictions.csv"

# Models and reports
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODEL_PATH = SAVED_MODELS_DIR / "best_model.keras"
BASELINE_MODEL_PATH = SAVED_MODELS_DIR / "baseline_model.keras"
FINAL_MODEL_PATH = SAVED_MODELS_DIR / "asirra_cats_dogs_final.keras"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
METRICS_DIR = REPORTS_DIR / "metrics"
STEP_REPORTS_DIR = REPORTS_DIR / "steps"
TEST_REPORTS_DIR = REPORTS_DIR / "tests"
PROJECT_SUMMARY_PATH = REPORTS_DIR / "PROJECT_SUMMARY.md"
TEST_RESULTS_JSON = TEST_REPORTS_DIR / "test_results.json"
TEST_RESULTS_MD = TEST_REPORTS_DIR / "test_results.md"
FINAL_TEST_METRICS_PATH = METRICS_DIR / "final_test.json"
TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.json"
VALIDATION_SUMMARY_PATH = METRICS_DIR / "validation_summary.json"
CLASS_BALANCE_CHART_PATH = EDA_DIR / "class_balance_train.png"
DIMENSION_DISTRIBUTION_PATH = EDA_DIR / "dimension_distribution_train.png"
SPLIT_COUNTS_CHART_PATH = EDA_DIR / "split_counts.png"
TRAINING_CURVES_PATH = METRICS_DIR / "training_curves.png"
TRAINING_PROGRESS_PATH = METRICS_DIR / "training_progress_epoch1.png"
VAL_CONFUSION_MATRIX_PATH = METRICS_DIR / "val_confusion_matrix.png"
VAL_MISCLASSIFIED_GRID_PATH = METRICS_DIR / "val_misclassified_grid.png"
STRATIFICATION_CHART_PATH = EDA_DIR / "stratification_ratios.png"
DEDUP_SUMMARY_PATH = EDA_DIR / "dedup_summary.png"
DIMENSION_SCATTER_PATH = EDA_DIR / "dimension_scatter_train.png"
AUGMENTATION_PREVIEW_PATH = EDA_DIR / "augmentation_preview.png"
TRAIN_VAL_BAR_PATH = METRICS_DIR / "train_val_metrics_bar.png"
VAL_ROC_PATH = METRICS_DIR / "val_roc_curve.png"
VAL_PR_PATH = METRICS_DIR / "val_pr_curve.png"
VAL_CONFUSION_NORM_PATH = METRICS_DIR / "val_confusion_matrix_normalized.png"
VAL_CLASS_METRICS_PATH = METRICS_DIR / "val_per_class_metrics.png"
VAL_CONFIDENCE_HIST_PATH = METRICS_DIR / "val_confidence_histogram.png"
VAL_CORRECT_GRID_PATH = METRICS_DIR / "val_correct_high_conf_grid.png"
OPTIMIZED_TRAINING_HISTORY_PATH = METRICS_DIR / "optimized_training_history.json"
OPTIMIZED_VALIDATION_SUMMARY_PATH = METRICS_DIR / "optimized_validation_summary.json"
OPTIMIZED_TRAINING_CURVES_PATH = METRICS_DIR / "optimized_training_curves.png"
EARLY_STOPPING_SUMMARY_PATH = METRICS_DIR / "early_stopping_summary.png"

# Split
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Images
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 16

# Training
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Default to VGG16 transfer learning so full train split fits in ~8GB RAM (see specs optional extension).
# Set USE_TRANSFER_LEARNING=false to attempt the literal from-scratch VGG stack (needs GPU / 16GB+ RAM).
USE_TRANSFER_LEARNING = os.getenv("USE_TRANSFER_LEARNING", "true").lower() in ("1", "true", "yes")
TRANSFER_EPOCHS = 1

# Gate one-time test evaluation (override with ALLOW_TEST_EVAL=true)
ALLOW_TEST_EVAL = os.getenv("ALLOW_TEST_EVAL", "false").lower() in ("1", "true", "yes")

PROCESSED_CLASS_DIRS = {
    "train": (PROCESSED_TRAIN_DIR / "cat", PROCESSED_TRAIN_DIR / "dog"),
    "val": (PROCESSED_VAL_DIR / "cat", PROCESSED_VAL_DIR / "dog"),
    "test": (PROCESSED_TEST_DIR / "cat", PROCESSED_TEST_DIR / "dog"),
}

LEAKAGE_TEST_DESCRIPTIONS = {
    "test_manifest_exists_after_split": (
        "Manifest exists with valid split column",
        "Ensures pipeline metadata is complete before training",
    ),
    "test_split_disjointness": (
        "No image_id appears in more than one split",
        "Prevents the same image from being in train and test",
    ),
    "test_content_hash_not_shared_across_splits": (
        "Duplicate files stay in a single split",
        "Prevents duplicate-image leakage across splits",
    ),
    "test_stratification_within_tolerance": (
        "Cat/dog ratio is similar across train, val, and test",
        "Prevents biased splits that distort evaluation",
    ),
    "test_processed_paths_under_processed_dir": (
        "Train/val/test paths live under data/processed/",
        "Prevents accidental training from raw data paths",
    ),
    "test_kaggle_test_dir_not_in_manifest": (
        "Unlabeled Kaggle test images are excluded from manifest",
        "Keeps unlabeled data out of labeled split logic",
    ),
    "test_splits_json_matches_manifest": (
        "splits.json counts match manifest.csv",
        "Ensures reporting and materialized splits are consistent",
    ),
    "test_evaluate_gate_blocks_test_by_default": (
        "Test evaluation is blocked unless ALLOW_TEST_EVAL=true",
        "Prevents accidental repeated test peeking",
    ),
}
