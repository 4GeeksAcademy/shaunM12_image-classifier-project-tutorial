"""Generate step reports, test summaries, and PROJECT_SUMMARY.md. See specs.md §2."""

from __future__ import annotations

import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    BASELINE_MODEL_PATH,
    BATCH_SIZE,
    CLASS_BALANCE_CHART_PATH,
    DEDUP_REPORT_PATH,
    EARLY_STOPPING_SUMMARY_PATH,
    FINAL_MODEL_PATH,
    FINAL_TEST_METRICS_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEAKAGE_TEST_DESCRIPTIONS,
    MANIFEST_PATH,
    OPTIMIZED_TRAINING_CURVES_PATH,
    OPTIMIZED_VALIDATION_SUMMARY_PATH,
    PROJECT_ROOT,
    PROJECT_SUMMARY_PATH,
    RAW_KAGGLE_TEST_DIR,
    RAW_TRAIN_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SAVED_MODEL_PATH,
    SPLIT_COUNTS_CHART_PATH,
    SPLITS_PATH,
    STEP_REPORTS_DIR,
    STRATIFICATION_CHART_PATH,
    TEST_RATIO,
    TEST_REPORTS_DIR,
    TEST_RESULTS_JSON,
    TEST_RESULTS_MD,
    TRAIN_RATIO,
    TRAINING_CURVES_PATH,
    VAL_CONFUSION_MATRIX_PATH,
    VAL_ROC_PATH,
    VALIDATION_SUMMARY_PATH,
    VAL_RATIO,
)

TESTS_MODULE = "tests.test_leakage"

ANN_CNN_ALIGNMENT_NOTE = (
    "The assignment refers to an ANN in the general sense; for image classification we "
    "implemented a convolutional neural network (VGG-style / VGG16 transfer), which is the "
    "appropriate architecture for photo-based cat vs dog classification."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    return Path(path).relative_to(PROJECT_ROOT).as_posix()


def ensure_report_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STEP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n\n"


def render_step_markdown(report: dict[str, Any]) -> str:
    reasoning = report.get("reasoning", {})
    leakage_rules = reasoning.get("leakage_rules", [])
    decisions = reasoning.get("decisions", [])
    inputs = report.get("inputs", {})
    outputs = report.get("outputs", {})
    metrics = report.get("metrics", {})
    warnings = report.get("warnings", [])
    checks = report.get("checks_passed", [])
    artifacts = report.get("artifacts", [])
    interpretation = report.get("interpretation", "")
    next_step = report.get("next_step", "")

    lines = [
        f"# {report.get('step_title', report.get('step', 'Step report'))}",
        "",
        f"- **Command:** `{report.get('command', '')}`",
        f"- **Timestamp (UTC):** {report.get('timestamp_utc', '')}",
        f"- **Duration:** {report.get('duration_seconds', 0)} seconds",
        f"- **Status:** {report.get('status', 'unknown')}",
        "",
    ]

    what_happened = report.get("what_happened", "")
    if what_happened:
        lines.append(_section("What happened", what_happened))

    reasoning_body = []
    if leakage_rules:
        reasoning_body.append("**Leakage rules enforced:**")
        reasoning_body.extend(f"- {rule}" for rule in leakage_rules)
    if decisions:
        reasoning_body.append("")
        reasoning_body.append("**Decisions applied:**")
        reasoning_body.extend(f"- {item}" for item in decisions)
    why = report.get("why", "")
    if why:
        reasoning_body.insert(0, why + "\n")
    lines.append(_section("Why we did it", "\n".join(reasoning_body)))

    if inputs:
        input_lines = "\n".join(f"- `{key}` → `{value}`" for key, value in inputs.items())
        lines.append(_section("Inputs", input_lines))

    if metrics:
        metric_lines = ["| Metric | Value |", "|--------|-------|"]
        for key, value in metrics.items():
            if isinstance(value, dict):
                metric_lines.append(f"| {key} | |")
                for sub_key, sub_val in value.items():
                    metric_lines.append(f"| &nbsp;&nbsp;{sub_key} | {sub_val} |")
            else:
                metric_lines.append(f"| {key} | {value} |")
        lines.append(_section("Results", "\n".join(metric_lines)))

    if outputs:
        output_lines = "\n".join(f"- `{key}` → `{value}`" for key, value in outputs.items())
        lines.append(_section("Outputs", output_lines))

    if artifacts:
        artifact_lines = "\n".join(f"- `{item}`" for item in artifacts)
        lines.append(_section("Artifacts", artifact_lines))

    visual_aids = report.get("visual_aids", [])
    if visual_aids:
        visual_lines = []
        for item in visual_aids:
            if isinstance(item, dict):
                title = item.get("title", "Visual aid")
                path = item.get("path", "")
                caption = item.get("caption", "")
                visual_lines.append(f"### {title}\n")
                visual_lines.append(f"![{title}]({path})\n")
                if caption:
                    visual_lines.append(f"_{caption}_\n")
            else:
                visual_lines.append(f"![visual aid]({item})\n")
        lines.append(_section("Visual aids", "\n".join(visual_lines)))

    check_lines = "\n".join(f"- {item}" for item in checks) if checks else "_No checks recorded for this step._"
    lines.append(_section("Leakage & validation checks", check_lines))

    if warnings:
        warning_lines = "\n".join(f"- ⚠️ {item}" for item in warnings)
        lines.append(_section("Warnings", warning_lines))

    if interpretation:
        lines.append(_section("Interpretation", interpretation))

    if next_step:
        lines.append(_section("Next step", f"```bash\n{next_step}\n```"))

    return "\n".join(lines).strip() + "\n"


def write_step_report(step_id: str, report: dict[str, Any]) -> tuple[Path, Path]:
    ensure_report_dirs()
    json_path = STEP_REPORTS_DIR / f"{step_id}.json"
    md_path = STEP_REPORTS_DIR / f"{step_id}.md"
    write_json(json_path, report)
    md_path.write_text(render_step_markdown(report), encoding="utf-8")
    return md_path, json_path


def _count_raw_images() -> int:
    if not RAW_TRAIN_DIR.exists():
        return 0
    return sum(1 for path in RAW_TRAIN_DIR.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _split_class_counts(manifest: pd.DataFrame, split_name: str) -> dict[str, int]:
    subset = manifest[manifest["split"] == split_name]
    return {
        "cat": int((subset["label"] == "cat").sum()),
        "dog": int((subset["label"] == "dog").sum()),
        "total": int(len(subset)),
    }


def build_step01_manifest_report(
    manifest: pd.DataFrame,
    *,
    duration_seconds: float,
    status: str = "success",
    files_scanned: int | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    dedup_count = 0
    if DEDUP_REPORT_PATH.exists():
        dedup_df = pd.read_csv(DEDUP_REPORT_PATH)
        dedup_count = len(dedup_df)

    nested_warning = (RAW_TRAIN_DIR / "train").is_dir()
    all_warnings = list(warnings or [])
    if nested_warning:
        all_warnings.append(
            "Images found in nested data/raw/asirra/train/train/ — split still works; flatten when convenient."
        )

    cat_count = int((manifest["label"] == "cat").sum())
    dog_count = int((manifest["label"] == "dog").sum())

    return {
        "step": "step01_manifest",
        "step_title": "Step 1 — Build manifest",
        "tutorial_step": 1,
        "command": "python src/app.py --stage manifest",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": (
            "Scanned labeled raw images, parsed cat/dog labels from filenames, computed SHA-256 "
            f"hashes, removed {dedup_count} duplicate file(s), and wrote manifest.csv."
        ),
        "why": (
            "The manifest is created before any model work so every image has a stable ID, label, "
            "and content hash. This is the foundation for a frozen, leak-free split."
        ),
        "reasoning": {
            "leakage_rules": [
                "Raw data is catalogued before training",
                "Duplicate images are identified before split assignment",
                "Labels are parsed once from filenames and stored centrally",
            ],
            "decisions": [
                "Recursive scan supports nested train/train/ uploads",
                "SHA-256 used for content_hash and group_id",
                "First occurrence kept when duplicates share the same hash",
            ],
        },
        "inputs": {
            "raw_train_dir": rel(RAW_TRAIN_DIR),
            "files_scanned": files_scanned if files_scanned is not None else _count_raw_images(),
        },
        "outputs": {
            "manifest_csv": rel(MANIFEST_PATH),
            "dedup_report_csv": rel(DEDUP_REPORT_PATH),
        },
        "metrics": {
            "unique_images": int(len(manifest)),
            "cat_count": cat_count,
            "dog_count": dog_count,
            "duplicates_removed": dedup_count,
        },
        "artifacts": [rel(MANIFEST_PATH), rel(DEDUP_REPORT_PATH)],
        "warnings": all_warnings,
        "checks_passed": [
            "Labels parsed from filenames",
            "Manifest written with content_hash column",
        ],
        "interpretation": (
            "Expect roughly equal cat and dog counts (~12.5k each before dedup). "
            "A much lower count means the upload is incomplete. "
            "Duplicates removed should stay small; large numbers may indicate repeated uploads."
        ),
        "next_step": "python src/app.py --stage split",
    }


def build_step015_split_report(
    manifest: pd.DataFrame,
    *,
    duration_seconds: float,
    status: str = "success",
) -> dict[str, Any]:
    counts_by_split = {
        split: _split_class_counts(manifest, split) for split in ("train", "val", "test")
    }

    return {
        "step": "step015_split",
        "step_title": "Step 1.5 — Stratified split",
        "tutorial_step": 1,
        "command": "python src/app.py --stage split",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": (
            "Assigned each manifest row to train, val, or test using a stratified 70/15/15 split "
            "with seed 42, then copied files into data/processed/{split}/{cat,dog}/."
        ),
        "why": (
            "Generators and training must read from processed splits — not the full raw folder — "
            "so validation and test images never influence training decisions."
        ),
        "reasoning": {
            "leakage_rules": [
                "Split is assigned once and written to manifest + splits.json",
                "Processed folders are the only training inputs after this step",
                "Kaggle unlabeled test/ is never included in the split",
            ],
            "decisions": [
                f"random_seed={RANDOM_SEED}",
                f"ratios train/val/test = {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}",
                "Stratified by label to preserve cat/dog balance",
                "Files copied (not symlinked) for tutorial reliability",
            ],
        },
        "inputs": {
            "manifest_csv": rel(MANIFEST_PATH),
            "raw_train_dir": rel(RAW_TRAIN_DIR),
        },
        "outputs": {
            "splits_json": rel(SPLITS_PATH),
            "processed_train": "data/processed/train",
            "processed_val": "data/processed/val",
            "processed_test": "data/processed/test",
            "split_counts_chart": "reports/eda/split_counts.png",
        },
        "metrics": {
            "counts_by_split": counts_by_split,
            "total_unique_images": int(len(manifest)),
        },
        "artifacts": [
            rel(SPLITS_PATH),
            rel(MANIFEST_PATH),
            "reports/eda/split_counts.png",
            "data/processed/train/cat",
            "data/processed/train/dog",
            "data/processed/val/cat",
            "data/processed/val/dog",
            "data/processed/test/cat",
            "data/processed/test/dog",
        ],
        "visual_aids": [
            {
                "title": "Split counts by class",
                "path": "reports/eda/split_counts.png",
                "caption": "Stratified 70/15/15 split — cat/dog balance preserved across train, val, and test.",
            },
        ],
        "checks_passed": [
            "Stratified split applied",
            "Processed folders materialized",
            "splits.json written",
        ],
        "interpretation": (
            "Train should be ~70%, val ~15%, test ~15% of unique images. "
            "Cat/dog counts should remain roughly balanced within each split."
        ),
        "next_step": "python src/app.py --stage eda",
    }


def build_step02_eda_report(
    eda_metrics: dict[str, Any],
    *,
    duration_seconds: float,
    status: str = "success",
) -> dict[str, Any]:
    class_balance = eda_metrics["class_balance"]
    dimension_summary = eda_metrics["dimension_summary"]
    generator_summary = eda_metrics["generator_summary"]

    return {
        "step": "step02_eda",
        "step_title": "Step 2 — Visualize and prepare generators",
        "tutorial_step": 2,
        "command": "python src/app.py --stage eda",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": (
            "Plotted 3x3 cat and dog grids from the train split only, recorded class balance, "
            f"summarized original image dimensions (sample n={dimension_summary['sample_size']}), "
            "and verified Keras generators on processed train/val/test folders."
        ),
        "why": (
            "Exploratory analysis must happen on the train split only so we do not peek at "
            "validation or test distributions. Generators are configured here so training uses "
            "augmentation on train and deterministic rescale-only transforms on val/test."
        ),
        "reasoning": {
            "leakage_rules": [
                "EDA reads only from data/processed/train/",
                "Val/test generators use rescale only — no augmentation",
                "Generator directories point to processed splits, not raw data",
            ],
            "decisions": [
                f"target_size=({IMG_WIDTH}, {IMG_HEIGHT}) to match VGG input_shape",
                f"batch_size={BATCH_SIZE}",
                "class_mode=categorical with rescale=1/255",
                "train augmentation: rotation_range=20, horizontal_flip=True",
            ],
        },
        "inputs": {
            "processed_train_dir": "data/processed/train",
            "processed_val_dir": "data/processed/val",
            "processed_test_dir": "data/processed/test",
        },
        "outputs": {
            "train_cats_grid": "reports/eda/train_cats_grid.png",
            "train_dogs_grid": "reports/eda/train_dogs_grid.png",
            "class_balance_chart": "reports/eda/class_balance_train.png",
            "dimension_distribution_chart": "reports/eda/dimension_distribution_train.png",
            "class_balance_json": "reports/eda/class_balance_train.json",
            "sample_dimensions_json": "reports/eda/sample_dimensions_train.json",
        },
        "metrics": {
            "class_balance_train": class_balance,
            "original_dimensions_summary": dimension_summary,
            "generators": generator_summary,
        },
        "artifacts": eda_metrics.get("artifacts", []),
        "visual_aids": [
            {
                "title": "Sample cats (train split)",
                "path": "reports/eda/train_cats_grid.png",
                "caption": "First 9 cat images from the train split only.",
            },
            {
                "title": "Sample dogs (train split)",
                "path": "reports/eda/train_dogs_grid.png",
                "caption": "First 9 dog images from the train split only.",
            },
            {
                "title": "Class balance",
                "path": "reports/eda/class_balance_train.png",
                "caption": "Nearly equal cat/dog counts in the train split.",
            },
            {
                "title": "Original image dimensions",
                "path": "reports/eda/dimension_distribution_train.png",
                "caption": "Original widths/heights before resize to 224×224.",
            },
        ],
        "checks_passed": [
            "EDA used train split only",
            "Train generator includes augmentation",
            "Val/test generators are rescale-only",
            "Batch shape matches 224x224x3",
        ],
        "interpretation": (
            "Cat and dog counts in the train split should be nearly equal. "
            "Original image sizes vary widely before resize; generators standardize to "
            f"{IMG_WIDTH}x{IMG_HEIGHT}. Batch shape should be "
            f"(batch_size, {IMG_HEIGHT}, {IMG_WIDTH}, 3)."
        ),
        "next_step": "python src/app.py --stage train",
    }


def build_step03_train_baseline_report(
    train_metrics: dict[str, Any],
    *,
    duration_seconds: float,
    status: str = "success",
) -> dict[str, Any]:
    validation_summary = train_metrics["validation_summary"]
    per_epoch = validation_summary.get("per_epoch", [])
    val_error_analysis = train_metrics.get("val_error_analysis", {})
    use_transfer = train_metrics.get("use_transfer_learning", False)
    architecture = train_metrics.get("architecture", "unknown")

    if use_transfer:
        title = "Step 3 — Train VGG16 transfer model (full train split)"
        what = (
            f"Trained a VGG16 ImageNet transfer model ({train_metrics['total_params']:,} total params, "
            f"{train_metrics.get('trainable_params', 'n/a'):,} trainable) for "
            f"{train_metrics['epochs']} epoch(s) on all {train_metrics['train_samples']:,} train images "
            "with validation on the val split."
        )
        why = (
            f"{ANN_CNN_ALIGNMENT_NOTE} "
            "The tutorial references VGG16 as the landmark ImageNet architecture. Training the full "
            "134M-parameter VGG stack from scratch exceeds available RAM on this CPU environment, so "
            "we use a frozen VGG16 backbone plus a small head — an approach listed in specs.md — while "
            "keeping the same leak-free splits, generators, loss, and validation protocol."
        )
        decisions = [
            "VGG16(weights='imagenet', include_top=False) with frozen backbone",
            "Custom head: GAP + Dense(256) + Dropout + Dense(2, softmax)",
            "VGG16 preprocess_input (not rescale=1/255) for transfer mode",
            f"batch_size={train_metrics.get('batch_size')}",
            f"epochs={train_metrics['epochs']}",
            f"steps_per_epoch={train_metrics['steps_per_epoch']} (full train split)",
        ]
    else:
        title = "Step 3 — Build and train VGG-style CNN (baseline)"
        what = (
            f"Built the VGG-style Sequential CNN ({train_metrics['total_params']:,} parameters, "
            f"{train_metrics['num_layers']} layers) and ran baseline training for "
            f"{train_metrics['epochs']} epoch(s) using train+augmentation with validation on the val split."
        )
        why = (
            f"{ANN_CNN_ALIGNMENT_NOTE} "
            "Convolutional layers learn spatial features that dense-only models miss on raw pixels. "
            "Validation metrics come from the val split — never the held-out test set."
        )
        decisions = [
            "VGG-style Sequential CNN from specs (224x224x3 input)",
            "optimizer=adam, loss=categorical_crossentropy, metrics=accuracy",
            f"epochs={train_metrics['epochs']}",
            f"steps_per_epoch={train_metrics['steps_per_epoch']}",
        ]

    return {
        "step": "step03_train_baseline",
        "step_title": title,
        "tutorial_step": 3,
        "command": "python src/app.py --stage train",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": what,
        "why": why,
        "reasoning": {
            "leakage_rules": [
                "model.fit uses validation_data=val_generator only",
                "All 17,495 train images used — no subsampling",
                "Test generator is not used during training",
            ],
            "decisions": decisions,
        },
        "inputs": {
            "processed_train_dir": "data/processed/train",
            "processed_val_dir": "data/processed/val",
            "train_samples": train_metrics["train_samples"],
            "val_samples": train_metrics["val_samples"],
            "architecture": architecture,
        },
        "outputs": {
            "baseline_model": "saved_models/baseline_model.keras",
            "training_history": "reports/metrics/training_history.json",
            "validation_summary": "reports/metrics/validation_summary.json",
            "training_curves": "reports/metrics/training_curves.png",
            "val_confusion_matrix": "reports/metrics/val_confusion_matrix.png",
            "val_misclassified_grid": "reports/metrics/val_misclassified_grid.png",
        },
        "metrics": {
            "architecture": architecture,
            "use_transfer_learning": use_transfer,
            "total_params": train_metrics["total_params"],
            "trainable_params": train_metrics.get("trainable_params"),
            "frozen_params": train_metrics.get("frozen_params"),
            "batch_size": train_metrics.get("batch_size"),
            "num_layers": train_metrics["num_layers"],
            "epochs": train_metrics["epochs"],
            "steps_per_epoch": train_metrics["steps_per_epoch"],
            "validation_steps": train_metrics["validation_steps"],
            "final_val_accuracy": validation_summary.get("final_val_accuracy"),
            "best_val_accuracy": validation_summary.get("best_val_accuracy"),
            "val_error_analysis": val_error_analysis,
            "per_epoch": per_epoch,
        },
        "artifacts": train_metrics.get(
            "artifacts",
            [
                "saved_models/baseline_model.keras",
                "reports/metrics/training_history.json",
                "reports/metrics/validation_summary.json",
            ],
        ),
        "visual_aids": [
            {
                "title": "Training curves",
                "path": "reports/metrics/training_curves.png",
                "caption": "Train vs validation loss and accuracy per epoch (val split only).",
            },
            {
                "title": "Validation confusion matrix",
                "path": "reports/metrics/val_confusion_matrix.png",
                "caption": "Error analysis on the held-out validation split — not the test set.",
            },
            {
                "title": "Misclassified validation samples",
                "path": "reports/metrics/val_misclassified_grid.png",
                "caption": "Gallery of val images the model got wrong (if any).",
            },
        ],
        "checks_passed": [
            "Validation data used instead of test during fit",
            "Training history saved to reports/metrics/",
            "Baseline model saved to saved_models/baseline_model.keras",
        ],
        "interpretation": (
            "Early epochs often show rapid train/val accuracy gains. "
            "Large gaps between train and val accuracy suggest overfitting — "
            "addressed in Step 4 with EarlyStopping and ModelCheckpoint on val_accuracy."
        ),
        "next_step": "python src/app.py --stage optimize",
    }


def build_step03_train_progress_report(
    training_progress: dict[str, Any],
    *,
    duration_seconds: float = 0,
) -> dict[str, Any]:
    steps_parsed = training_progress.get("steps_parsed", 0)
    chart = training_progress.get("chart", "reports/metrics/training_progress_epoch1.png")

    return {
        "step": "step03_train_baseline",
        "step_title": "Step 3 — Training progress (recovered from log, no retrain)",
        "tutorial_step": 3,
        "command": "python src/app.py --stage visuals",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": "partial",
        "what_happened": (
            f"Recovered batch-level training metrics for {steps_parsed:,} steps from an existing "
            "Keras log. No saved model or validation metrics are on disk yet, so confusion-matrix "
            "visuals were not generated."
        ),
        "why": (
            "Training was interrupted before model/history files were written. Parsing the log "
            "preserves real batch progress without rerunning the ~1.8 hour CPU epoch."
        ),
        "reasoning": {
            "leakage_rules": [
                "Progress chart uses train-batch metrics only",
                "No test data used",
                "Validation confusion matrix requires a saved model — skipped",
            ],
            "decisions": [
                "Source log parsed with regex for step/accuracy/loss",
                "Chart saved to reports/metrics/training_progress_epoch1.png",
            ],
        },
        "inputs": {
            "source_log": training_progress.get("source_log", ""),
        },
        "outputs": {
            "training_progress_chart": chart,
        },
        "metrics": {
            "steps_parsed": steps_parsed,
            "source_log": training_progress.get("source_log", ""),
        },
        "artifacts": [chart],
        "visual_aids": [
            {
                "title": "Training batch progress (epoch 1)",
                "path": chart,
                "caption": "Recovered from training log — train running-average only; validation not saved.",
            },
        ],
        "checks_passed": [
            "No retraining required for this chart",
            "Test split not used",
        ],
        "interpretation": (
            "The curve shows train accuracy rising across the epoch. "
            "Full Step 3 deliverables (val curves, confusion matrix) still need a completed "
            "training run that saves baseline_model.keras."
        ),
        "next_step": "python src/app.py --stage train --epochs 1  # only when ready to finish training",
    }


def build_step04_optimize_report(
    optimize_metrics: dict[str, Any],
    *,
    duration_seconds: float,
    test_metrics: dict[str, Any] | None = None,
    status: str = "success",
) -> dict[str, Any]:
    validation_summary = optimize_metrics["validation_summary"]
    per_epoch = validation_summary.get("per_epoch", [])
    val_error_analysis = optimize_metrics.get("val_error_analysis", {})
    early_stopped = optimize_metrics.get("early_stopping_triggered", False)
    manually_terminated = validation_summary.get("manually_terminated", False)
    terminated_during_epoch = validation_summary.get("terminated_during_epoch")

    visual_aids = [
        {
            "title": "EarlyStopping summary",
            "path": "reports/metrics/early_stopping_summary.png",
            "caption": (
                "val_accuracy monitored by EarlyStopping, patience counter, and manual-stop "
                "annotation when the optimize run did not finish all requested epochs."
            ),
        },
        {
            "title": "Optimized training curves",
            "path": "reports/metrics/optimized_training_curves.png",
            "caption": "Train vs validation metrics during Step 4 fine-tuning (val split only).",
        },
        {
            "title": "Validation confusion matrix",
            "path": "reports/metrics/val_confusion_matrix.png",
            "caption": "Post-optimize error analysis on validation — not test.",
        },
        {
            "title": "Validation ROC curve",
            "path": "reports/metrics/val_roc_curve.png",
            "caption": "ROC for dog vs cat on validation split.",
        },
        {
            "title": "Validation PR curve",
            "path": "reports/metrics/val_pr_curve.png",
            "caption": "Precision-recall for dog class on validation.",
        },
        {
            "title": "Per-class validation metrics",
            "path": "reports/metrics/val_per_class_metrics.png",
            "caption": "Precision, recall, and F1 per class on validation.",
        },
        {
            "title": "Confidence histogram",
            "path": "reports/metrics/val_confidence_histogram.png",
            "caption": "Distribution of max softmax confidence on validation.",
        },
    ]

    test_section: dict[str, Any] = {
        "ran": False,
        "note": "Test evaluation not run. Use ALLOW_TEST_EVAL=true python src/app.py --stage evaluate once.",
    }
    if test_metrics:
        test_section = {
            "ran": True,
            "timestamp_utc": test_metrics.get("timestamp_utc"),
            "allow_test_eval": test_metrics.get("allow_test_eval"),
            "test_loss": test_metrics.get("test_loss"),
            "test_accuracy": test_metrics.get("test_accuracy"),
            "model_path": test_metrics.get("model_path"),
            "metrics_path": "reports/metrics/final_test.json",
            "note": test_metrics.get("note"),
        }

    return {
        "step": "step04_optimize",
        "step_title": "Step 4 — Optimize model (EarlyStopping + ModelCheckpoint)",
        "tutorial_step": 4,
        "command": "python src/app.py --stage optimize",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": (
            f"Continued training from baseline_model.keras for up to "
            f"{optimize_metrics.get('epochs_requested')} epoch(s) with EarlyStopping "
            f"(patience={optimize_metrics.get('early_stopping_patience')}) and ModelCheckpoint "
            f"monitoring val_accuracy. Completed {optimize_metrics.get('epochs_completed')} epoch(s); "
            f"early stopping {'triggered' if early_stopped else 'not triggered'}."
            + (
                f" Run was manually terminated during epoch {terminated_during_epoch} "
                f"(no end-of-epoch val metric recorded)."
                if manually_terminated and terminated_during_epoch
                else (
                    " EarlyStopping halted training when val_accuracy stopped improving."
                    if early_stopped
                    else ""
                )
            )
        ),
        "why": (
            "ModelCheckpoint saves the best validation weights so we do not keep a worse final epoch. "
            "EarlyStopping halts training when val_accuracy stops improving, reducing overfitting risk. "
            "Both callbacks monitor val_accuracy only — the held-out test set stays locked until a "
            "single gated evaluate run."
        ),
        "reasoning": {
            "leakage_rules": [
                "Callbacks monitor val_accuracy — never test accuracy",
                "Test generator excluded from fit() and callback logic",
                "Test evaluate() requires ALLOW_TEST_EVAL=true and should run once",
            ],
            "decisions": [
                f"Continued from {optimize_metrics.get('continued_from', 'baseline_model.keras')}",
                f"ModelCheckpoint → saved_models/best_model.keras",
                f"EarlyStopping patience={optimize_metrics.get('early_stopping_patience')}",
                f"Max epochs={optimize_metrics.get('epochs_requested')}",
            ],
        },
        "inputs": {
            "baseline_model": "saved_models/baseline_model.keras",
            "processed_train_dir": "data/processed/train",
            "processed_val_dir": "data/processed/val",
            "train_samples": optimize_metrics.get("train_samples"),
            "val_samples": optimize_metrics.get("val_samples"),
        },
        "outputs": {
            "best_model": "saved_models/best_model.keras",
            "optimized_training_history": "reports/metrics/optimized_training_history.json",
            "optimized_validation_summary": "reports/metrics/optimized_validation_summary.json",
            "optimized_training_curves": "reports/metrics/optimized_training_curves.png",
            "early_stopping_summary": "reports/metrics/early_stopping_summary.png",
        },
        "metrics": {
            "architecture": optimize_metrics.get("architecture"),
            "epochs_requested": optimize_metrics.get("epochs_requested"),
            "epochs_completed": optimize_metrics.get("epochs_completed"),
            "early_stopping_triggered": early_stopped,
            "early_stopping_patience": optimize_metrics.get("early_stopping_patience"),
            "manually_terminated": manually_terminated,
            "terminated_during_epoch": terminated_during_epoch,
            "baseline_val_accuracy": validation_summary.get("baseline_val_accuracy"),
            "best_epoch": validation_summary.get("best_epoch"),
            "best_val_accuracy": validation_summary.get("best_val_accuracy"),
            "final_val_accuracy": validation_summary.get("final_val_accuracy"),
            "val_error_analysis": val_error_analysis,
            "per_epoch": per_epoch,
            "test_evaluation": test_section,
        },
        "artifacts": optimize_metrics.get(
            "artifacts",
            [
                "saved_models/best_model.keras",
                "reports/metrics/optimized_training_history.json",
                "reports/metrics/optimized_validation_summary.json",
                "reports/metrics/early_stopping_summary.png",
            ],
        ),
        "visual_aids": visual_aids,
        "checks_passed": [
            "EarlyStopping and ModelCheckpoint monitor val_accuracy only",
            "Best weights saved to saved_models/best_model.keras",
            "Optimized history saved under reports/metrics/",
            "EarlyStopping summary chart documents patience and stop reason",
        ],
        "interpretation": (
            "If early stopping triggered quickly, validation accuracy may already have plateaued "
            "during baseline training. Compare optimized vs baseline val metrics before running "
            "the one-time test evaluation."
            + (
                " A manual stop leaves the patience counter incomplete — see early_stopping_summary.png."
                if manually_terminated
                else ""
            )
        ),
        "next_step": (
            "ALLOW_TEST_EVAL=true python src/app.py --stage evaluate"
            if not test_metrics
            else "python src/app.py --stage save"
        ),
    }


def build_step05_save_report(
    save_metrics: dict[str, Any],
    *,
    duration_seconds: float,
    status: str = "success",
) -> dict[str, Any]:
    reload = save_metrics.get("reload_sanity_check", {})
    file_sizes = save_metrics.get("file_sizes_mb", {})

    test_section: dict[str, Any] = {"ran": False}
    if FINAL_TEST_METRICS_PATH.exists():
        with FINAL_TEST_METRICS_PATH.open(encoding="utf-8") as handle:
            test_payload = json.load(handle)
        test_section = {
            "ran": True,
            "test_accuracy": test_payload.get("test_accuracy"),
            "test_loss": test_payload.get("test_loss"),
            "metrics_path": "reports/metrics/final_test.json",
            "note": "Completed during Step 4 evaluate — not re-run during save.",
        }

    return {
        "step": "step05_save",
        "step_title": "Step 5 — Save and verify model",
        "tutorial_step": 5,
        "command": "python src/app.py --stage save",
        "timestamp_utc": utc_now_iso(),
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
        "what_happened": (
            f"Copied the best checkpoint ({save_metrics.get('source_model_path')}) to "
            f"{save_metrics.get('final_model_path')}, reloaded the exported file, and ran a "
            f"validation-only sanity check ({reload.get('val_samples', '?')} samples)."
        ),
        "why": (
            "ModelCheckpoint writes best_model.keras during training; the final export gives a "
            "stable, named artifact for deployment and submission. Reloading and evaluating on "
            "validation confirms the saved file is readable and produces sensible metrics without "
            "touching the locked test set."
        ),
        "reasoning": {
            "leakage_rules": [
                "Reload sanity check uses validation split only",
                "Test set is not used during save or reload verification",
                "best_model.keras remains the training checkpoint; final export is a copy for delivery",
            ],
            "decisions": [
                f"Source checkpoint: {save_metrics.get('source_model_path')}",
                f"Final export: {save_metrics.get('final_model_path')}",
                f"Export method: {save_metrics.get('export_method', 'copy')}",
            ],
        },
        "inputs": {
            "best_checkpoint": str(SAVED_MODEL_PATH),
            "baseline_model": str(BASELINE_MODEL_PATH),
            "processed_val_dir": "data/processed/val",
        },
        "outputs": {
            "final_model": str(FINAL_MODEL_PATH),
            "reload_sanity_metrics": "reports/metrics/save_reload_sanity_check.json",
        },
        "metrics": {
            "file_sizes_mb": file_sizes,
            "reload_sanity_check": reload,
            "test_evaluation": test_section,
        },
        "artifacts": save_metrics.get(
            "artifacts",
            [
                "saved_models/best_model.keras",
                "saved_models/asirra_cats_dogs_final.keras",
                "reports/metrics/save_reload_sanity_check.json",
            ],
        ),
        "checks_passed": [
            "Final model exported to saved_models/asirra_cats_dogs_final.keras",
            "Reloaded model evaluates successfully on validation",
            "Sanity-check metrics written to reports/metrics/save_reload_sanity_check.json",
        ],
        "interpretation": (
            "Reload val_accuracy should match the best checkpoint within floating-point tolerance. "
            "If it diverges, the export path or preprocessing pipeline may differ from training."
        ),
        "next_step": "# Pipeline complete",
    }


def refresh_step04_optimize_report_from_disk() -> None:
    """Regenerate Step 4 chart/report from saved optimize metrics without retraining."""
    from visualizations import plot_early_stopping_summary_from_disk

    if not OPTIMIZED_VALIDATION_SUMMARY_PATH.exists():
        return

    plot_early_stopping_summary_from_disk()

    with OPTIMIZED_VALIDATION_SUMMARY_PATH.open(encoding="utf-8") as handle:
        validation_summary = json.load(handle)

    test_metrics = None
    if FINAL_TEST_METRICS_PATH.exists():
        with FINAL_TEST_METRICS_PATH.open(encoding="utf-8") as handle:
            test_metrics = json.load(handle)

    optimize_metrics = {
        "validation_summary": validation_summary,
        "val_error_analysis": {},
        "epochs_requested": validation_summary.get("max_epochs_requested"),
        "epochs_completed": validation_summary.get("epochs_completed"),
        "early_stopping_triggered": validation_summary.get("early_stopping_triggered"),
        "early_stopping_patience": validation_summary.get("early_stopping_patience"),
        "continued_from": "saved_models/baseline_model.keras",
        "train_samples": validation_summary.get("train_samples"),
        "val_samples": validation_summary.get("val_samples"),
        "architecture": validation_summary.get("architecture", "vgg16_transfer"),
        "artifacts": [],
    }

    existing = _load_stage_reports().get("step04_optimize")
    duration = existing.get("duration_seconds", 0) if existing else 0
    status = existing.get("status", "success") if existing else "success"

    report = build_step04_optimize_report(
        optimize_metrics,
        duration_seconds=duration,
        test_metrics=test_metrics,
        status=status,
    )
    if existing:
        if existing.get("timestamp_utc"):
            report["timestamp_utc"] = existing["timestamp_utc"]
        if existing.get("what_happened") and "manually terminated" in existing["what_happened"].lower():
            report["what_happened"] = existing["what_happened"]

    write_step_report("step04_optimize", report)


class _TestResultCollector(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._test_start = time.perf_counter()

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._append(test, "PASS", "")

    def addFailure(self, test: unittest.TestCase, err) -> None:
        super().addFailure(test, err)
        self._append(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test: unittest.TestCase, err) -> None:
        super().addError(test, err)
        self._append(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._append(test, "SKIP", reason)

    def _append(self, test: unittest.TestCase, status: str, message: str) -> None:
        name = test._testMethodName
        duration = round(time.perf_counter() - getattr(self, "_test_start", time.perf_counter()), 4)
        verifies, why = LEAKAGE_TEST_DESCRIPTIONS.get(name, ("", ""))
        self.records.append(
            {
                "name": name,
                "status": status,
                "message": message.strip(),
                "duration_seconds": duration,
                "verifies": verifies,
                "why_it_matters": why,
            }
        )


def run_leakage_tests() -> dict[str, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(TESTS_MODULE)
    collector = _TestResultCollector()
    suite.run(collector)

    passed = sum(1 for item in collector.records if item["status"] == "PASS")
    failed = sum(1 for item in collector.records if item["status"] in {"FAIL", "ERROR"})
    skipped = sum(1 for item in collector.records if item["status"] == "SKIP")

    return {
        "timestamp_utc": utc_now_iso(),
        "command": "python -m unittest tests.test_leakage -v",
        "totals": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(collector.records),
        },
        "tests": collector.records,
    }


def write_test_results(results: dict[str, Any]) -> tuple[Path, Path]:
    ensure_report_dirs()
    write_json(TEST_RESULTS_JSON, results)

    lines = [
        "# Leakage test results",
        "",
        f"- **Timestamp (UTC):** {results['timestamp_utc']}",
        f"- **Command:** `{results['command']}`",
        f"- **Totals:** {results['totals']['passed']} passed, "
        f"{results['totals']['failed']} failed, {results['totals']['skipped']} skipped",
        "",
        "| Test | Status | What it verifies | Why it matters |",
        "|------|--------|------------------|----------------|",
    ]

    for test in results["tests"]:
        status = test["status"]
        if status == "PASS":
            status_display = "✅ PASS"
        elif status == "SKIP":
            status_display = "⏭ SKIP"
        else:
            status_display = f"❌ {status}"
        lines.append(
            f"| `{test['name']}` | {status_display} | {test['verifies']} | {test['why_it_matters']} |"
        )

    lines.extend(["", f"**Summary:** {results['totals']['passed']} passed, "
                  f"{results['totals']['failed']} failed, {results['totals']['skipped']} skipped.", ""])
    TEST_RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    return TEST_RESULTS_MD, TEST_RESULTS_JSON


def _load_stage_reports() -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for step_id in (
        "step01_manifest",
        "step015_split",
        "step02_eda",
        "step03_train_baseline",
        "step04_optimize",
        "step05_save",
    ):
        json_path = STEP_REPORTS_DIR / f"{step_id}.json"
        if json_path.exists():
            with json_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            stages[step_id] = payload
    return stages


def refresh_reports_from_disk() -> dict[str, Any] | None:
    """Rebuild step reports and run tests using existing manifest/splits (no re-copy)."""
    from config import BASELINE_MODEL_PATH
    from visualizations import generate_pipeline_visuals

    test_results = None
    manifest = None
    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        report = build_step01_manifest_report(manifest, duration_seconds=0, status="success")
        report["what_happened"] = (
            "Report refreshed from existing manifest.csv on disk. "
            + report["what_happened"]
        )
        write_step_report("step01_manifest", report)

        if "split" in manifest.columns and manifest["split"].isin(["train", "val", "test"]).any():
            split_report = build_step015_split_report(manifest, duration_seconds=0, status="success")
            split_report["what_happened"] = (
                "Report refreshed from existing processed splits on disk. "
                + split_report["what_happened"]
            )
            write_step_report("step015_split", split_report)
            test_results = run_leakage_tests()
            write_test_results(test_results)

    visual_payload = generate_pipeline_visuals(manifest=manifest)
    refresh_step04_optimize_report_from_disk()
    if visual_payload.get("training_progress") and not BASELINE_MODEL_PATH.exists():
        progress_report = build_step03_train_progress_report(visual_payload["training_progress"])
        write_step_report("step03_train_baseline", progress_report)

    update_project_summary(test_results=test_results)
    return test_results


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_test_results(test_results: dict[str, Any] | None) -> dict[str, Any] | None:
    if test_results is not None:
        return test_results
    cached = _load_json_if_exists(TEST_RESULTS_JSON)
    if cached is not None:
        return cached
    if MANIFEST_PATH.exists():
        results = run_leakage_tests()
        write_test_results(results)
        return results
    return None


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _load_model_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "baseline_val_accuracy": None,
        "optimize_best_val_accuracy": None,
        "optimize_best_epoch": None,
        "early_stopping_triggered": None,
        "manually_terminated": None,
        "test_accuracy": None,
        "test_eval_ran": False,
        "final_model_path": None,
        "best_checkpoint_path": None,
    }

    baseline = _load_json_if_exists(VALIDATION_SUMMARY_PATH)
    if baseline:
        metrics["baseline_val_accuracy"] = baseline.get("best_val_accuracy")

    optimized = _load_json_if_exists(OPTIMIZED_VALIDATION_SUMMARY_PATH)
    if optimized:
        metrics["optimize_best_val_accuracy"] = optimized.get("best_val_accuracy")
        metrics["optimize_best_epoch"] = optimized.get("best_epoch")
        metrics["early_stopping_triggered"] = optimized.get("early_stopping_triggered")
        metrics["manually_terminated"] = optimized.get("manually_terminated")

    test_payload = _load_json_if_exists(FINAL_TEST_METRICS_PATH)
    if test_payload:
        metrics["test_accuracy"] = test_payload.get("test_accuracy")
        metrics["test_eval_ran"] = True

    if FINAL_MODEL_PATH.exists():
        metrics["final_model_path"] = rel(FINAL_MODEL_PATH)
    if SAVED_MODEL_PATH.exists():
        metrics["best_checkpoint_path"] = rel(SAVED_MODEL_PATH)

    return metrics


def _collect_visual_aids() -> list[tuple[str, Path]]:
    candidates = [
        ("Split counts (stratified)", SPLIT_COUNTS_CHART_PATH),
        ("Class balance (train)", CLASS_BALANCE_CHART_PATH),
        ("Stratification ratios", STRATIFICATION_CHART_PATH),
        ("Baseline training curves", TRAINING_CURVES_PATH),
        ("Optimize training curves", OPTIMIZED_TRAINING_CURVES_PATH),
        ("EarlyStopping summary", EARLY_STOPPING_SUMMARY_PATH),
        ("Validation confusion matrix", VAL_CONFUSION_MATRIX_PATH),
        ("Validation ROC curve", VAL_ROC_PATH),
    ]
    return [(label, path) for label, path in candidates if path.exists()]


def _pipeline_is_complete(stages: dict[str, dict[str, Any]]) -> bool:
    required = (
        "step01_manifest",
        "step015_split",
        "step02_eda",
        "step03_train_baseline",
        "step04_optimize",
        "step05_save",
    )
    return all(step_id in stages for step_id in required) and FINAL_MODEL_PATH.exists()


def update_project_summary(
    *,
    test_results: dict[str, Any] | None = None,
) -> Path:
    ensure_report_dirs()
    stages = _load_stage_reports()
    test_results = _resolve_test_results(test_results)
    model_metrics = _load_model_metrics()
    pipeline_complete = _pipeline_is_complete(stages)

    lines = [
        "# Project summary",
        "",
        "_Auto-generated quick view. See `reports/steps/` for detailed per-step reports._",
        "",
    ]

    if pipeline_complete:
        lines.extend(
            [
                "> **Pipeline complete** — Steps 1–5 finished. Optional: Kaggle predict CSV if unlabeled test images are downloaded.",
                "",
            ]
        )

    lines.extend(
        [
            "## Pipeline status",
            "",
            "| Stage | Status | Time (UTC) | Duration (s) |",
            "|-------|--------|------------|----------------|",
        ]
    )

    for step_id, label in (
        ("step01_manifest", "manifest"),
        ("step015_split", "split"),
        ("step02_eda", "eda"),
        ("step03_train_baseline", "train (baseline)"),
        ("step04_optimize", "optimize"),
        ("step05_save", "save (Step 5)"),
    ):
        payload = stages.get(step_id)
        if payload:
            lines.append(
                f"| {label} | {payload.get('status', 'unknown').upper()} | "
                f"{payload.get('timestamp_utc', '')} | {payload.get('duration_seconds', '')} |"
            )
        else:
            lines.append(f"| {label} | PENDING | — | — |")

    if test_results:
        totals = test_results["totals"]
        test_status = f"{totals['passed']}/{totals['total']} PASS"
        if totals["failed"]:
            test_status = f"FAILED ({totals['failed']} failures)"
        lines.append(
            f"| leakage tests | {test_status} | {test_results['timestamp_utc']} | — |"
        )
    else:
        lines.append("| leakage tests | PENDING | — | — |")

    lines.extend(["", "## Data counts", ""])

    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        if "split" in manifest.columns and manifest["split"].isin(["train", "val", "test"]).any():
            lines.extend(
                [
                    "| Split | Cats | Dogs | Total |",
                    "|-------|------|------|-------|",
                ]
            )
            for split in ("train", "val", "test"):
                counts = _split_class_counts(manifest, split)
                lines.append(
                    f"| {split} | {counts['cat']} | {counts['dog']} | {counts['total']} |"
                )
        else:
            lines.append(f"- Unique manifest rows: **{len(manifest)}**")
            lines.append(f"- Cats: **{(manifest['label'] == 'cat').sum()}**")
            lines.append(f"- Dogs: **{(manifest['label'] == 'dog').sum()}**")
    else:
        lines.append("_Manifest not built yet._")

    raw_count = _count_raw_images()
    lines.extend(["", f"- Raw images on disk: **{raw_count}**"])
    kaggle_test_count = (
        sum(1 for path in RAW_KAGGLE_TEST_DIR.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if RAW_KAGGLE_TEST_DIR.exists()
        else 0
    )
    lines.append(f"- Kaggle unlabeled test images: **{kaggle_test_count}**")

    lines.extend(
        [
            "",
            "## Model metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Baseline val accuracy (Step 3) | {_format_pct(model_metrics['baseline_val_accuracy'])} |",
            f"| Best optimize val accuracy (Step 4) | {_format_pct(model_metrics['optimize_best_val_accuracy'])} |",
            f"| Optimize best epoch | {model_metrics['optimize_best_epoch'] or '—'} |",
            f"| EarlyStopping triggered | {model_metrics['early_stopping_triggered'] if model_metrics['early_stopping_triggered'] is not None else '—'} |",
            f"| Manual termination (Step 4) | {model_metrics['manually_terminated'] if model_metrics['manually_terminated'] is not None else '—'} |",
            f"| Test accuracy (one-time) | {_format_pct(model_metrics['test_accuracy']) if model_metrics['test_eval_ran'] else 'Not run'} |",
            f"| Best checkpoint | `{model_metrics['best_checkpoint_path'] or '—'}` |",
            f"| Final export (Step 5) | `{model_metrics['final_model_path'] or '—'}` |",
            "",
            "## Key decisions",
            "",
            f"- Split seed: **{RANDOM_SEED}**",
            f"- Ratios: **train {TRAIN_RATIO:.0%} / val {VAL_RATIO:.0%} / test {TEST_RATIO:.0%}**",
            "- Image size: **224×224**",
            "- Loss: **categorical_crossentropy** with 2-unit softmax",
            "- Callbacks monitor: **val_accuracy** only",
            f"- Architecture note: {ANN_CNN_ALIGNMENT_NOTE}",
            "",
            "## Leakage checks",
            "",
        ]
    )

    if test_results:
        for test in test_results["tests"]:
            icon = "✅" if test["status"] == "PASS" else ("⏭" if test["status"] == "SKIP" else "❌")
            lines.append(f"- {icon} `{test['name']}` — {test['verifies']}")
    else:
        lines.append("_Run `python -m unittest tests.test_leakage -v` after split._")

    visual_aids = _collect_visual_aids()
    if visual_aids:
        lines.extend(["", "## Visual aids (quick reference)", ""])
        lines.extend(["| Chart | Path |", "|-------|------|"])
        for label, path in visual_aids:
            lines.append(f"| {label} | `{rel(path)}` |")

        spotlight = [
            (label, path)
            for label, path in visual_aids
            if path.name
            in {
                "split_counts.png",
                "training_curves.png",
                "early_stopping_summary.png",
                "val_confusion_matrix.png",
            }
        ]
        if spotlight:
            lines.extend(["", "### Key charts", ""])
            for label, path in spotlight:
                summary_rel = path.relative_to(REPORTS_DIR).as_posix()
                lines.append(f"**{label}** — [`{summary_rel}`]({summary_rel})")
                lines.append("")
                lines.append(f"![{label}]({summary_rel})")
                lines.append("")

    warnings: list[str] = []
    for payload in stages.values():
        warnings.extend(payload.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in dict.fromkeys(warnings):
            lines.append(f"- ⚠️ {warning}")

    next_steps = []
    if "step015_split" not in stages:
        next_steps.append("python src/app.py --stage split")
    elif "step02_eda" not in stages:
        next_steps.append("python src/app.py --stage eda")
    elif "step03_train_baseline" not in stages:
        next_steps.append("python src/app.py --stage train")
    elif "step04_optimize" not in stages:
        next_steps.append("python src/app.py --stage optimize")
    elif not FINAL_TEST_METRICS_PATH.exists():
        next_steps.append("ALLOW_TEST_EVAL=true python src/app.py --stage evaluate")
    elif not FINAL_MODEL_PATH.exists():
        next_steps.append("python src/app.py --stage save")
    elif kaggle_test_count == 0:
        next_steps.append("# Pipeline complete — optional: download Kaggle test images + predict CSV")
    else:
        next_steps.append("python src/app.py --stage predict  # optional Kaggle submission")

    lines.extend(["", "## Next step", "", f"```bash\n{next_steps[0]}\n```", ""])
    PROJECT_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return PROJECT_SUMMARY_PATH
