"""Exploratory data analysis on the train split only. See specs.md Step 2."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import (
    BATCH_SIZE,
    EDA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    PROCESSED_TRAIN_DIR,
    PROJECT_ROOT,
)
from data.generators import build_generators
from visualizations import plot_class_balance, plot_dimension_distribution

CLASS_BALANCE_PATH = EDA_DIR / "class_balance_train.json"
TRAIN_CATS_GRID_PATH = EDA_DIR / "train_cats_grid.png"
TRAIN_DOGS_GRID_PATH = EDA_DIR / "train_dogs_grid.png"
SAMPLE_DIMENSIONS_PATH = EDA_DIR / "sample_dimensions_train.json"
GRID_SAMPLE_COUNT = 9
DIMENSION_SAMPLE_COUNT = 100


def _list_train_images(label: str) -> list[Path]:
    directory = PROCESSED_TRAIN_DIR / label
    if not directory.exists():
        raise FileNotFoundError(
            f"Train split not found at {directory}. Run: python src/app.py --stage split"
        )
    return sorted(path for path in directory.glob("*.jpg") if path.is_file())


def _plot_image_grid(image_paths: list[Path], title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(title, fontsize=14)

    for axis, path in zip(axes.flat, image_paths[:GRID_SAMPLE_COUNT]):
        image = Image.open(path).convert("RGB")
        axis.imshow(image)
        axis.set_title(path.name, fontsize=8)
        axis.axis("off")

    for axis in axes.flat[len(image_paths[:GRID_SAMPLE_COUNT]) :]:
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _sample_dimensions(image_paths: list[Path], sample_size: int) -> list[dict]:
    rng = random.Random(42)
    sample = image_paths if len(image_paths) <= sample_size else rng.sample(image_paths, sample_size)
    rows: list[dict] = []
    for path in sample:
        with Image.open(path) as image:
            width, height = image.size
        rows.append(
            {
                "filename": path.name,
                "original_width": width,
                "original_height": height,
                "target_width": IMG_WIDTH,
                "target_height": IMG_HEIGHT,
            }
        )
    return rows


def _summarize_dimensions(rows: list[dict]) -> dict:
    widths = [row["original_width"] for row in rows]
    heights = [row["original_height"] for row in rows]
    return {
        "sample_size": len(rows),
        "original_width_min": int(min(widths)),
        "original_width_max": int(max(widths)),
        "original_height_min": int(min(heights)),
        "original_height_max": int(max(heights)),
        "original_width_mean": round(float(np.mean(widths)), 1),
        "original_height_mean": round(float(np.mean(heights)), 1),
        "target_size": [IMG_WIDTH, IMG_HEIGHT],
    }


def run_eda(*, write_grids: bool = True) -> dict:
    """Run Step 2 EDA on processed train split only and return metrics for reporting."""
    cat_paths = _list_train_images("cat")
    dog_paths = _list_train_images("dog")

    if len(cat_paths) < GRID_SAMPLE_COUNT or len(dog_paths) < GRID_SAMPLE_COUNT:
        raise ValueError("Not enough train images to build 3x3 EDA grids.")

    if write_grids:
        _plot_image_grid(
            cat_paths[:GRID_SAMPLE_COUNT],
            "First 9 cats (train split only)",
            TRAIN_CATS_GRID_PATH,
        )
        _plot_image_grid(
            dog_paths[:GRID_SAMPLE_COUNT],
            "First 9 dogs (train split only)",
            TRAIN_DOGS_GRID_PATH,
        )

    class_balance = {
        "split": "train",
        "source_dir": str(PROCESSED_TRAIN_DIR.relative_to(PROJECT_ROOT)),
        "cats": len(cat_paths),
        "dogs": len(dog_paths),
        "total": len(cat_paths) + len(dog_paths),
        "cat_ratio": round(len(cat_paths) / (len(cat_paths) + len(dog_paths)), 4),
    }
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    with CLASS_BALANCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(class_balance, handle, indent=2)

    dimension_rows = _sample_dimensions(cat_paths + dog_paths, DIMENSION_SAMPLE_COUNT)
    dimension_summary = _summarize_dimensions(dimension_rows)
    with SAMPLE_DIMENSIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {"summary": dimension_summary, "samples": dimension_rows[:10]},
            handle,
            indent=2,
        )

    class_balance_chart = plot_class_balance(cats=len(cat_paths), dogs=len(dog_paths))
    dimension_chart = plot_dimension_distribution(
        dimension_rows,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
    )

    generators = build_generators()
    batch = next(generators.train)
    images, labels = batch
    generator_summary = {
        "train_samples": generators.train.samples,
        "val_samples": generators.val.samples,
        "test_samples": generators.test.samples,
        "class_indices": generators.train.class_indices,
        "batch_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "steps_per_epoch": generators.steps_per_epoch,
        "validation_steps": generators.validation_steps,
        "test_steps": generators.test_steps,
        "batch_size": BATCH_SIZE,
        "augmentation_on_train_only": True,
    }

    return {
        "class_balance": class_balance,
        "dimension_summary": dimension_summary,
        "generator_summary": generator_summary,
        "artifacts": [
            str(TRAIN_CATS_GRID_PATH.relative_to(PROJECT_ROOT)),
            str(TRAIN_DOGS_GRID_PATH.relative_to(PROJECT_ROOT)),
            class_balance_chart,
            dimension_chart,
            str(CLASS_BALANCE_PATH.relative_to(PROJECT_ROOT)),
            str(SAMPLE_DIMENSIONS_PATH.relative_to(PROJECT_ROOT)),
        ],
    }


if __name__ == "__main__":
    metrics = run_eda()
    print(json.dumps(metrics, indent=2))
