"""Stratified split and materialize data/processed/. See specs.md Step 1.5."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    MANIFEST_PATH,
    PROCESSED_CLASS_DIRS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLITS_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data.manifest import build_manifest


def _clear_processed_images() -> None:
    for split_dirs in PROCESSED_CLASS_DIRS.values():
        for class_dir in split_dirs:
            class_dir.mkdir(parents=True, exist_ok=True)
            for path in class_dir.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink()


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    if not manifest["split"].eq("").all() and manifest["split"].notna().all():
        return manifest

    train_df, temp_df = train_test_split(
        manifest,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=manifest["label"],
    )

    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    manifest = manifest.copy()
    manifest["split"] = ""
    manifest.loc[train_df.index, "split"] = "train"
    manifest.loc[val_df.index, "split"] = "val"
    manifest.loc[test_df.index, "split"] = "test"
    return manifest


def materialize_split(manifest: pd.DataFrame) -> dict:
    _clear_processed_images()

    counts = {"train": 0, "val": 0, "test": 0, "cat": 0, "dog": 0}

    for _, row in manifest.iterrows():
        split = row["split"]
        label = row["label"]
        src = PROJECT_ROOT / row["filepath"]
        dest_dir = PROJECT_ROOT / "data/processed" / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / row["filename"]

        if not src.exists():
            raise FileNotFoundError(f"Source image missing: {src}")

        shutil.copy2(src, dest)
        counts[split] += 1
        counts[label] += 1

    return counts


def write_splits_json(manifest: pd.DataFrame, counts: dict) -> None:
    summary = {
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "total_images": int(len(manifest)),
        "counts_by_split": {
            split: int((manifest["split"] == split).sum()) for split in ("train", "val", "test")
        },
        "counts_by_label": {
            label: int((manifest["label"] == label).sum()) for label in ("cat", "dog")
        },
        "copied_files": counts,
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLITS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_split(rebuild_manifest: bool = False) -> tuple[pd.DataFrame, dict]:
    if rebuild_manifest or not MANIFEST_PATH.exists():
        manifest, _ = build_manifest()
    else:
        manifest = pd.read_csv(MANIFEST_PATH)

    manifest = assign_splits(manifest)
    counts = materialize_split(manifest)
    manifest.to_csv(MANIFEST_PATH, index=False)
    write_splits_json(manifest, counts)

    print(f"Split complete. Summary written to {SPLITS_PATH}")
    split_counts = {split: int((manifest["split"] == split).sum()) for split in ("train", "val", "test")}
    print("Counts:", split_counts)

    metadata = {"counts_by_split": split_counts, "copied_files": counts}
    return manifest, metadata


if __name__ == "__main__":
    run_split(rebuild_manifest=True)
