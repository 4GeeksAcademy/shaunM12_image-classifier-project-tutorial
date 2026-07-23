"""Charts and grids for pipeline reports. See specs.md Steps 2–4."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config import (
    AUGMENTATION_PREVIEW_PATH,
    BASELINE_MODEL_PATH,
    BATCH_SIZE,
    CLASS_BALANCE_CHART_PATH,
    DEDUP_REPORT_PATH,
    DEDUP_SUMMARY_PATH,
    DIMENSION_DISTRIBUTION_PATH,
    DIMENSION_SCATTER_PATH,
    EARLY_STOPPING_SUMMARY_PATH,
    EDA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    MANIFEST_PATH,
    METRICS_DIR,
    OPTIMIZED_TRAINING_CURVES_PATH,
    OPTIMIZED_TRAINING_HISTORY_PATH,
    OPTIMIZED_VALIDATION_SUMMARY_PATH,
    PROCESSED_TRAIN_DIR,
    PROJECT_ROOT,
    SAVED_MODEL_PATH,
    SPLIT_COUNTS_CHART_PATH,
    STRATIFICATION_CHART_PATH,
    TRAIN_VAL_BAR_PATH,
    TRAINING_CURVES_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_PROGRESS_PATH,
    VALIDATION_SUMMARY_PATH,
    VAL_CLASS_METRICS_PATH,
    VAL_CONFIDENCE_HIST_PATH,
    VAL_CONFUSION_MATRIX_PATH,
    VAL_CONFUSION_NORM_PATH,
    VAL_CORRECT_GRID_PATH,
    VAL_MISCLASSIFIED_GRID_PATH,
    VAL_PR_PATH,
    VAL_ROC_PATH,
)

SAMPLE_DIMENSIONS_PATH = EDA_DIR / "sample_dimensions_train.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _class_names_from_indices(class_indices: dict[str, int]) -> list[str]:
    return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def plot_class_balance(
    *,
    cats: int,
    dogs: int,
    output_path: Path = CLASS_BALANCE_CHART_PATH,
    title: str = "Class balance (train split only)",
) -> str:
    _ensure_parent(output_path)
    labels = ["Cats", "Dogs"]
    counts = [cats, dogs]

    fig, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, counts, color=["#4C72B0", "#DD8452"])
    axis.set_ylabel("Image count")
    axis.set_title(title)
    axis.set_ylim(0, max(counts) * 1.15)

    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_dimension_distribution(
    dimension_rows: list[dict],
    *,
    output_path: Path = DIMENSION_DISTRIBUTION_PATH,
    target_size: tuple[int, int],
) -> str:
    _ensure_parent(output_path)
    widths = [row["original_width"] for row in dimension_rows]
    heights = [row["original_height"] for row in dimension_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].axvline(target_size[0], color="#C44E52", linestyle="--", label=f"target {target_size[0]}px")
    axes[0].set_title("Original width distribution")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(heights, bins=20, color="#55A868", edgecolor="white")
    axes[1].axvline(target_size[1], color="#C44E52", linestyle="--", label=f"target {target_size[1]}px")
    axes[1].set_title("Original height distribution")
    axes[1].set_xlabel("Pixels")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("Original image sizes before resize to 224×224", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_split_counts(
    counts_by_split: dict[str, dict[str, int]],
    *,
    output_path: Path = SPLIT_COUNTS_CHART_PATH,
) -> str:
    _ensure_parent(output_path)
    splits = ["train", "val", "test"]
    cats = [counts_by_split[split]["cat"] for split in splits]
    dogs = [counts_by_split[split]["dog"] for split in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - width / 2, cats, width, label="Cats", color="#4C72B0")
    axis.bar(x + width / 2, dogs, width, label="Dogs", color="#DD8452")
    axis.set_xticks(x, splits)
    axis.set_ylabel("Image count")
    axis.set_title("Stratified split counts (70 / 15 / 15)")
    axis.legend()

    for index, split in enumerate(splits):
        total = cats[index] + dogs[index]
        axis.text(index, max(cats[index], dogs[index]) + 150, f"{total:,}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_split_counts_from_manifest(manifest: pd.DataFrame) -> str:
    counts_by_split: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split_name]
        counts_by_split[split_name] = {
            "cat": int((subset["label"] == "cat").sum()),
            "dog": int((subset["label"] == "dog").sum()),
            "total": int(len(subset)),
        }
    return plot_split_counts(counts_by_split)


def plot_training_curves(
    history: dict[str, list[float]],
    *,
    output_path: Path = TRAINING_CURVES_PATH,
    title: str = "Baseline training curves (validation on val split only)",
) -> str:
    _ensure_parent(output_path)
    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.get("loss", []), marker="o", label="train loss")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], marker="o", label="val loss")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("accuracy", []), marker="o", label="train accuracy")
    if history.get("val_accuracy"):
        axes[1].plot(epochs, history["val_accuracy"], marker="o", label="val accuracy")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def _patience_counter(val_accuracies: list[float]) -> list[int]:
    """Simulate EarlyStopping wait counter from completed epoch val_accuracy values."""
    best = float("-inf")
    waits: list[int] = []
    for value in val_accuracies:
        if value > best:
            best = value
            waits.append(0)
        else:
            waits.append((waits[-1] + 1) if waits else 1)
    return waits


def plot_early_stopping_summary(
    validation_summary: dict[str, Any],
    *,
    baseline_val_accuracy: float | None = None,
    output_path: Path = EARLY_STOPPING_SUMMARY_PATH,
) -> str | None:
    """Plot val_accuracy timeline with EarlyStopping metadata (no retraining required)."""
    per_epoch = validation_summary.get("per_epoch") or []
    optimize_vals = [
        float(row["val_accuracy"])
        for row in per_epoch
        if row.get("val_accuracy") is not None
    ]
    if not optimize_vals and validation_summary.get("best_val_accuracy") is not None:
        optimize_vals = [float(validation_summary["best_val_accuracy"])]

    if baseline_val_accuracy is None:
        baseline_val_accuracy = validation_summary.get("baseline_val_accuracy")
    if baseline_val_accuracy is None and VALIDATION_SUMMARY_PATH.exists():
        with VALIDATION_SUMMARY_PATH.open(encoding="utf-8") as handle:
            baseline_summary = json.load(handle)
        baseline_val_accuracy = baseline_summary.get("best_val_accuracy")

    if baseline_val_accuracy is None and not optimize_vals:
        return None

    patience = int(validation_summary.get("early_stopping_patience", 0))
    max_epochs = int(validation_summary.get("max_epochs_requested", len(optimize_vals) or 1))
    epochs_completed = int(validation_summary.get("epochs_completed", len(optimize_vals)))
    early_stopping_triggered = bool(validation_summary.get("early_stopping_triggered", False))
    manually_terminated = bool(
        validation_summary.get("manually_terminated")
        or (
            not early_stopping_triggered
            and epochs_completed < max_epochs
        )
    )
    terminated_during_epoch = validation_summary.get("terminated_during_epoch")
    if manually_terminated and terminated_during_epoch is None:
        terminated_during_epoch = epochs_completed + 1

    best_val = float(
        validation_summary.get("best_val_accuracy")
        or (max(optimize_vals) if optimize_vals else baseline_val_accuracy)
    )
    best_epoch = int(validation_summary.get("best_epoch", optimize_vals.index(max(optimize_vals)) + 1 if optimize_vals else 0))

    x_points: list[float] = [0.0]
    y_points: list[float] = [float(baseline_val_accuracy or best_val)]
    labels = ["Baseline\n(checkpoint)"]

    for index, value in enumerate(optimize_vals, start=1):
        x_points.append(float(index))
        y_points.append(value)
        labels.append(f"Optimize\nepoch {index}")

    waits = _patience_counter(optimize_vals)
    patience_epochs = list(range(1, len(waits) + 1))

    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})

    ax_val = axes[0]
    ax_val.plot(x_points, y_points, marker="o", linewidth=2, color="#4C72B0", label="val_accuracy")
    ax_val.axhline(
        best_val,
        color="#55A868",
        linestyle="--",
        linewidth=1.5,
        label=f"Best in optimize run ({best_val:.4f})",
    )

    if baseline_val_accuracy is not None:
        ax_val.axhline(
            float(baseline_val_accuracy),
            color="#DD8452",
            linestyle=":",
            linewidth=1.5,
            label=f"Baseline Step 3 ({float(baseline_val_accuracy):.4f})",
        )

    for x_val, y_val, label in zip(x_points, y_points, labels):
        ax_val.annotate(f"{y_val:.4f}", (x_val, y_val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    if manually_terminated and terminated_during_epoch is not None:
        stop_x = float(terminated_during_epoch) - 0.5
        ax_val.axvline(stop_x, color="#C44E52", linestyle=":", linewidth=1.5)
        ax_val.text(
            stop_x,
            min(y_points) - 0.002,
            f"Manual stop\n(epoch {terminated_during_epoch} incomplete)",
            color="#C44E52",
            ha="center",
            va="top",
            fontsize=8,
        )
    elif early_stopping_triggered:
        ax_val.axvline(float(epochs_completed) + 0.15, color="#C44E52", linestyle=":", linewidth=1.5)
        ax_val.text(
            float(epochs_completed) + 0.15,
            min(y_points) - 0.002,
            "EarlyStopping\ntriggered",
            color="#C44E52",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax_val.set_xticks(x_points)
    ax_val.set_xticklabels(labels, fontsize=8)
    ax_val.set_ylabel("val_accuracy")
    ax_val.set_title("Validation accuracy monitored by EarlyStopping")
    ax_val.set_xlim(-0.3, max(max(x_points) + 0.8, float(max_epochs) * 0.35))
    ax_val.legend(loc="lower right", fontsize=8)
    ax_val.grid(alpha=0.3)

    ax_patience = axes[1]
    if patience_epochs:
        colors = ["#C44E52" if wait >= patience else "#4C72B0" for wait in waits]
        ax_patience.bar(patience_epochs, waits, color=colors)
        ax_patience.axhline(patience, color="#55A868", linestyle="--", linewidth=1.5, label=f"patience={patience}")
        ax_patience.set_xlabel("Optimize epoch")
        ax_patience.set_ylabel("Epochs without val improvement")
        ax_patience.set_title("EarlyStopping patience counter")
        ax_patience.set_ylim(0, max(patience + 1, max(waits) + 1))
        ax_patience.legend(fontsize=8)
        ax_patience.grid(axis="y", alpha=0.3)
    else:
        ax_patience.axis("off")
        ax_patience.text(0.5, 0.5, "No completed\noptimize epochs", ha="center", va="center")

    summary_lines = [
        f"Monitor: val_accuracy",
        f"Patience: {patience}",
        f"Max epochs requested: {max_epochs}",
        f"Epochs completed: {epochs_completed}",
        f"Best optimize epoch: {best_epoch}",
        f"EarlyStopping triggered: {'Yes' if early_stopping_triggered else 'No'}",
        f"Manual termination: {'Yes' if manually_terminated else 'No'}",
    ]
    if manually_terminated and terminated_during_epoch is not None:
        summary_lines.append(f"Stopped during epoch: {terminated_during_epoch}")

    fig.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F5F5", "edgecolor": "#CCCCCC"},
    )
    fig.suptitle("Step 4 — EarlyStopping summary (validation split only)", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_early_stopping_summary_from_disk(
    *,
    optimized_summary_path: Path = OPTIMIZED_VALIDATION_SUMMARY_PATH,
    output_path: Path = EARLY_STOPPING_SUMMARY_PATH,
) -> str | None:
    if not optimized_summary_path.exists():
        return None
    with optimized_summary_path.open(encoding="utf-8") as handle:
        validation_summary = json.load(handle)
    return plot_early_stopping_summary(validation_summary, output_path=output_path)


def plot_training_curves_from_json(
    history_path: Path,
    *,
    output_path: Path = TRAINING_CURVES_PATH,
) -> str | None:
    if not history_path.exists():
        return None
    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not history.get("loss"):
        return None
    return plot_training_curves(history, output_path=output_path)


_KERAS_PROGRESS_RE = re.compile(
    r"(\d+)/(\d+).*?accuracy: ([0-9.]+) - loss: ([0-9.]+)"
)


def parse_keras_progress_log(log_path: Path) -> list[dict[str, float | int]]:
    """Parse step-wise train metrics from a Keras fit() log (no retraining required)."""
    if not log_path.exists():
        return []

    latest_by_step: dict[int, dict[str, float | int]] = {}
    for match in _KERAS_PROGRESS_RE.finditer(log_path.read_text(encoding="utf-8", errors="ignore")):
        step = int(match.group(1))
        latest_by_step[step] = {
            "step": step,
            "total_steps": int(match.group(2)),
            "accuracy": float(match.group(3)),
            "loss": float(match.group(4)),
        }
    return [latest_by_step[step] for step in sorted(latest_by_step)]


def plot_training_progress_from_log(
    log_path: Path,
    *,
    output_path: Path = TRAINING_PROGRESS_PATH,
) -> str | None:
    """Plot batch-level train metrics recovered from an existing training log."""
    rows = parse_keras_progress_log(log_path)
    if len(rows) < 2:
        return None

    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    total_steps = int(rows[-1]["total_steps"])

    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, losses, color="#C44E52", linewidth=1.5)
    axes[0].set_title("Train loss (batch running average)")
    axes[0].set_xlabel(f"Training step (of {total_steps})")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, accuracies, color="#4C72B0", linewidth=1.5)
    axes[1].set_title("Train accuracy (batch running average)")
    axes[1].set_xlabel(f"Training step (of {total_steps})")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Recovered training progress from log (epoch 1 — validation/model not saved)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def _resolve_inference_model_path() -> Path | None:
    if SAVED_MODEL_PATH.exists():
        return SAVED_MODEL_PATH
    if BASELINE_MODEL_PATH.exists():
        return BASELINE_MODEL_PATH
    return None


def plot_stratification_ratios(
    manifest: pd.DataFrame,
    *,
    output_path: Path = STRATIFICATION_CHART_PATH,
) -> str:
    _ensure_parent(output_path)
    splits = ["train", "val", "test"]
    cat_ratios: list[float] = []
    totals: list[int] = []

    for split_name in splits:
        subset = manifest[manifest["split"] == split_name]
        cats = int((subset["label"] == "cat").sum())
        total = int(len(subset))
        cat_ratios.append(cats / total if total else 0.0)
        totals.append(total)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(splits, cat_ratios, color="#4C72B0")
    axes[0].axhline(0.5, color="#C44E52", linestyle="--", label="50% baseline")
    axes[0].set_ylim(0.45, 0.55)
    axes[0].set_ylabel("Cat ratio (cats / total)")
    axes[0].set_title("Class ratio per split (stratification check)")
    axes[0].legend()

    for index, split_name in enumerate(splits):
        axes[0].text(index, cat_ratios[index] + 0.002, f"{cat_ratios[index]:.3f}", ha="center", fontsize=9)

    x = np.arange(len(splits))
    width = 0.35
    cats = [int((manifest[manifest["split"] == split]["label"] == "cat").sum()) for split in splits]
    dogs = [totals[index] - cats[index] for index in range(len(splits))]
    axes[1].bar(x - width / 2, cats, width, label="Cats", color="#4C72B0")
    axes[1].bar(x + width / 2, dogs, width, label="Dogs", color="#DD8452")
    axes[1].set_xticks(x, splits)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Absolute counts per split")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_dedup_summary(*, output_path: Path = DEDUP_SUMMARY_PATH) -> str | None:
    if not DEDUP_REPORT_PATH.exists():
        return None

    dedup = pd.read_csv(DEDUP_REPORT_PATH)
    if dedup.empty:
        return None

    labels = dedup["duplicate_filepath"].str.contains("/cat.", regex=False).map({True: "cat", False: "dog"})
    counts = labels.value_counts().reindex(["cat", "dog"], fill_value=0)

    _ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    axis.set_ylabel("Duplicate files removed")
    axis.set_title(f"Deduplication summary ({len(dedup)} duplicates dropped)")
    for index, label in enumerate(counts.index):
        axis.text(index, counts.values[index], str(int(counts.values[index])), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_dimension_scatter(*, output_path: Path = DIMENSION_SCATTER_PATH) -> str | None:
    if not SAMPLE_DIMENSIONS_PATH.exists():
        return None

    with SAMPLE_DIMENSIONS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not rows:
        return None

    widths = [row["original_width"] for row in rows]
    heights = [row["original_height"] for row in rows]
    colors = []
    for row in rows:
        label = row.get("label")
        if label is None:
            label = "cat" if row.get("filename", "").startswith("cat.") else "dog"
        colors.append("#4C72B0" if label == "cat" else "#DD8452")

    _ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(widths, heights, c=colors, alpha=0.65, edgecolors="white", linewidth=0.3)
    axis.axvline(IMG_WIDTH, color="#C44E52", linestyle="--", label=f"target {IMG_WIDTH}px")
    axis.axhline(IMG_HEIGHT, color="#8172B2", linestyle="--", label=f"target {IMG_HEIGHT}px")
    axis.set_xlabel("Original width (px)")
    axis.set_ylabel("Original height (px)")
    axis.set_title("Original image dimensions (train sample)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_augmentation_preview(
    *,
    output_path: Path = AUGMENTATION_PREVIEW_PATH,
    use_vgg16_preprocess: bool = True,
) -> str:
    from data.generators import _make_datagen

    cat_dir = PROCESSED_TRAIN_DIR / "cat"
    sample_paths = sorted(cat_dir.glob("*.jpg"))
    if not sample_paths:
        raise FileNotFoundError(f"No train images found under {cat_dir}")

    image = Image.open(sample_paths[0]).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    array = np.asarray(image, dtype=np.float32)
    # Use rescale datagen for the preview grid so rotations/flips are human-visible.
    datagen = _make_datagen(augment=True, use_vgg16_preprocess=False)

    _ensure_parent(output_path)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle("Augmentation preview (train generator settings)", fontsize=12)

    for axis in axes.flat:
        batch = datagen.random_transform(array.copy())
        display = np.clip(batch * 255, 0, 255)
        axis.imshow(display.astype(np.uint8))
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def plot_train_val_metrics_bar(
    *,
    summary_path: Path = VALIDATION_SUMMARY_PATH,
    output_path: Path = TRAIN_VAL_BAR_PATH,
) -> str | None:
    if not summary_path.exists():
        return None

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    train_acc = summary.get("final_train_accuracy")
    val_acc = summary.get("final_val_accuracy")
    if train_acc is None or val_acc is None:
        return None

    _ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(5, 4))
    labels = ["Train accuracy", "Val accuracy"]
    values = [train_acc, val_acc]
    bars = axis.bar(labels, values, color=["#55A868", "#4C72B0"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Final train vs validation accuracy (baseline)")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path.relative_to(PROJECT_ROOT).as_posix()


def _collect_val_predictions(model, val_generator) -> dict[str, Any]:
    class_names = _class_names_from_indices(val_generator.class_indices)
    steps = math.ceil(val_generator.samples / val_generator.batch_size)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []

    val_generator.reset()
    for step in range(steps):
        batch_x, batch_y = val_generator[step]
        predictions = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y, axis=1).tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_proba.extend(predictions.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)
    confidences = y_proba_arr.max(axis=1)

    return {
        "class_names": class_names,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_proba": y_proba_arr,
        "confidences": confidences,
        "filepaths": [Path(path) for path in val_generator.filepaths],
    }


def plot_all_val_visuals(
    model,
    val_generator,
    *,
    confusion_path: Path = VAL_CONFUSION_MATRIX_PATH,
    misclassified_path: Path = VAL_MISCLASSIFIED_GRID_PATH,
    norm_confusion_path: Path = VAL_CONFUSION_NORM_PATH,
    roc_path: Path = VAL_ROC_PATH,
    pr_path: Path = VAL_PR_PATH,
    class_metrics_path: Path = VAL_CLASS_METRICS_PATH,
    confidence_hist_path: Path = VAL_CONFIDENCE_HIST_PATH,
    correct_grid_path: Path = VAL_CORRECT_GRID_PATH,
    max_gallery: int = 9,
) -> dict[str, Any]:
    """Full validation error analysis bundle — no retraining required."""
    for path in (
        confusion_path,
        misclassified_path,
        norm_confusion_path,
        roc_path,
        pr_path,
        class_metrics_path,
        confidence_hist_path,
        correct_grid_path,
    ):
        _ensure_parent(path)

    data = _collect_val_predictions(model, val_generator)
    class_names = data["class_names"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_proba = data["y_proba"]
    confidences = data["confidences"]
    filepaths = data["filepaths"]

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=120)
    plt.close(fig)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
        vmin=0,
        vmax=1,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Normalized validation confusion matrix")
    fig.tight_layout()
    fig.savefig(norm_confusion_path, dpi=120)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics_labels = class_names + ["macro avg"]
    precision = [report[name]["precision"] for name in class_names] + [report["macro avg"]["precision"]]
    recall = [report[name]["recall"] for name in class_names] + [report["macro avg"]["recall"]]
    f1 = [report[name]["f1-score"] for name in class_names] + [report["macro avg"]["f1-score"]]

    x = np.arange(len(metrics_labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    axis.bar(x, recall, width, label="Recall", color="#DD8452")
    axis.bar(x + width, f1, width, label="F1", color="#55A868")
    axis.set_xticks(x, metrics_labels, rotation=15)
    axis.set_ylim(0, 1.05)
    axis.set_title("Per-class validation metrics")
    axis.legend()
    fig.tight_layout()
    fig.savefig(class_metrics_path, dpi=120)
    plt.close(fig)

    dog_index = class_names.index("dog") if "dog" in class_names else 1
    y_binary = (y_true == dog_index).astype(int)
    y_score = y_proba[:, dog_index]
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, color="#4C72B0", label=f"ROC AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Validation ROC (positive class: dog)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    precision_curve, recall_curve, _ = precision_recall_curve(y_binary, y_score)
    pr_auc = auc(recall_curve, precision_curve)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall_curve, precision_curve, color="#DD8452", label=f"PR AUC = {pr_auc:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Validation precision-recall (positive class: dog)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(confidences, bins=20, color="#8172B2", edgecolor="white")
    axis.set_xlabel("Max softmax confidence")
    axis.set_ylabel("Count")
    axis.set_title("Validation prediction confidence distribution")
    fig.tight_layout()
    fig.savefig(confidence_hist_path, dpi=120)
    plt.close(fig)

    misclassified_indices = [index for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label != pred_label]
    gallery_indices = misclassified_indices[:max_gallery]
    gallery_paths = [filepaths[index] for index in gallery_indices]

    cols = 3
    rows = math.ceil(max(len(gallery_paths), 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3 * rows))
    axes_flat = np.atleast_1d(axes).flat

    if gallery_paths:
        for axis, path, index in zip(axes_flat, gallery_paths, gallery_indices):
            image = Image.open(path).convert("RGB")
            axis.imshow(image)
            true_name = class_names[y_true[index]]
            pred_name = class_names[y_pred[index]]
            axis.set_title(f"true={true_name}\npred={pred_name}", fontsize=8)
            axis.axis("off")
        for axis in axes_flat[len(gallery_paths) :]:
            axis.axis("off")
        mis_title = f"Misclassified validation samples (showing {len(gallery_paths)} of {len(misclassified_indices)})"
    else:
        for axis in axes_flat:
            axis.axis("off")
        axes_flat[0].text(0.5, 0.5, "No misclassifications on validation set", ha="center", va="center")
        mis_title = "Misclassified validation samples"

    fig.suptitle(mis_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(misclassified_path, dpi=120)
    plt.close(fig)

    correct_indices = [index for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) if true_label == pred_label]
    correct_indices.sort(key=lambda index: confidences[index], reverse=True)
    correct_gallery = correct_indices[:max_gallery]

    fig, axes = plt.subplots(rows, cols, figsize=(9, 3 * rows))
    axes_flat = np.atleast_1d(axes).flat
    for axis, index in zip(axes_flat, correct_gallery):
        image = Image.open(filepaths[index]).convert("RGB")
        axis.imshow(image)
        label_name = class_names[y_true[index]]
        axis.set_title(f"{label_name}\nconf={confidences[index]:.2f}", fontsize=8)
        axis.axis("off")
    for axis in axes_flat[len(correct_gallery) :]:
        axis.axis("off")
    fig.suptitle("High-confidence correct validation predictions", fontsize=11)
    fig.tight_layout()
    fig.savefig(correct_grid_path, dpi=120)
    plt.close(fig)

    accuracy = float(np.mean(y_true == y_pred))
    artifacts = [
        confusion_path.relative_to(PROJECT_ROOT).as_posix(),
        norm_confusion_path.relative_to(PROJECT_ROOT).as_posix(),
        misclassified_path.relative_to(PROJECT_ROOT).as_posix(),
        class_metrics_path.relative_to(PROJECT_ROOT).as_posix(),
        roc_path.relative_to(PROJECT_ROOT).as_posix(),
        pr_path.relative_to(PROJECT_ROOT).as_posix(),
        confidence_hist_path.relative_to(PROJECT_ROOT).as_posix(),
        correct_grid_path.relative_to(PROJECT_ROOT).as_posix(),
    ]
    return {
        "val_samples_evaluated": len(y_true),
        "val_accuracy_from_generator": round(accuracy, 4),
        "misclassified_count": len(misclassified_indices),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "artifacts": artifacts,
    }


def generate_pipeline_visuals(
    *,
    manifest: pd.DataFrame | None = None,
    training_log_path: Path | None = None,
    use_transfer: bool = True,
) -> dict[str, Any]:
    """Build all visuals that do not require retraining."""
    artifacts: list[str] = []
    payload: dict[str, Any] = {"artifacts": artifacts}

    if manifest is None and MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)

    if manifest is not None and "split" in manifest.columns:
        artifacts.append(plot_split_counts_from_manifest(manifest))
        artifacts.append(plot_stratification_ratios(manifest))

    dedup_chart = plot_dedup_summary()
    if dedup_chart:
        artifacts.append(dedup_chart)

    scatter_chart = plot_dimension_scatter()
    if scatter_chart:
        artifacts.append(scatter_chart)

    try:
        artifacts.append(plot_augmentation_preview(use_vgg16_preprocess=use_transfer))
    except FileNotFoundError:
        pass

    train_val_bar = plot_train_val_metrics_bar()
    if train_val_bar:
        artifacts.append(train_val_bar)

    curves = plot_training_curves_from_json(TRAINING_HISTORY_PATH)
    if curves:
        artifacts.append(curves)

    optimized_curves = plot_training_curves_from_json(
        OPTIMIZED_TRAINING_HISTORY_PATH,
        output_path=OPTIMIZED_TRAINING_CURVES_PATH,
    )
    if optimized_curves:
        artifacts.append(optimized_curves)

    early_stop_summary = plot_early_stopping_summary_from_disk()
    if early_stop_summary:
        artifacts.append(early_stop_summary)

    log_candidates = [
        training_log_path,
        Path("/tmp/train_log.txt"),
        Path("/tmp/train_log2.txt"),
    ]
    for candidate in log_candidates:
        if candidate and candidate.exists():
            progress = plot_training_progress_from_log(candidate)
            if progress:
                artifacts.append(progress)
                payload["training_progress"] = {
                    "source_log": str(candidate),
                    "steps_parsed": len(parse_keras_progress_log(candidate)),
                    "chart": progress,
                }
                break

    model_path = _resolve_inference_model_path()
    if model_path is not None:
        model_payload = regenerate_training_visuals(use_transfer=use_transfer, model_path=model_path)
        for path in model_payload.get("artifacts", []):
            if path not in artifacts:
                artifacts.append(path)
        payload.update({key: value for key, value in model_payload.items() if key != "artifacts"})

    return payload


def plot_val_error_analysis(
    model,
    val_generator,
    *,
    confusion_path: Path = VAL_CONFUSION_MATRIX_PATH,
    misclassified_path: Path = VAL_MISCLASSIFIED_GRID_PATH,
    max_gallery: int = 9,
) -> dict[str, Any]:
    """Backward-compatible wrapper around the full validation visual bundle."""
    return plot_all_val_visuals(
        model,
        val_generator,
        confusion_path=confusion_path,
        misclassified_path=misclassified_path,
        max_gallery=max_gallery,
    )


def regenerate_training_visuals(
    *,
    use_transfer: bool = True,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild training charts from saved history/model (e.g. after an older train run)."""
    from tensorflow.keras.models import load_model

    from data.generators import build_generators

    artifacts: list[str] = []
    payload: dict[str, Any] = {"artifacts": artifacts}

    curves = plot_training_curves_from_json(TRAINING_HISTORY_PATH)
    if curves:
        artifacts.append(curves)

    resolved_path = model_path or _resolve_inference_model_path()
    if resolved_path is None:
        return payload

    model = load_model(resolved_path)
    generators = build_generators(use_vgg16_preprocess=use_transfer)
    error_analysis = plot_all_val_visuals(model, generators.val)
    artifacts.extend(error_analysis["artifacts"])
    payload["val_error_analysis"] = error_analysis
    payload["model_used"] = resolved_path.relative_to(PROJECT_ROOT).as_posix()

    if VALIDATION_SUMMARY_PATH.exists():
        with VALIDATION_SUMMARY_PATH.open(encoding="utf-8") as handle:
            payload["validation_summary"] = json.load(handle)

    return payload
