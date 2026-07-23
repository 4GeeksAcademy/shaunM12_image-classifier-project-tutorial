"""Build manifest.csv from data/raw/asirra/train/. See specs.md Step 1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from config import (
    DEDUP_REPORT_PATH,
    INTERIM_DIR,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_TRAIN_DIR,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_label(filename: str) -> str | None:
    """Return cat or dog based on filename, or None if unrecognized."""
    name = filename.lower()
    if "cat" in name:
        return "cat"
    if "dog" in name:
        return "dog"
    return None


def hash_file(path: Path, chunk_size: int = 8192) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_image_paths(raw_train_dir: Path) -> list[Path]:
    """Find labeled images under raw train (supports nested train/train/ drops)."""
    if not raw_train_dir.exists():
        raise FileNotFoundError(f"Raw train directory not found: {raw_train_dir}")

    paths: list[Path] = []
    for path in sorted(raw_train_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if parse_label(path.name) is not None:
                paths.append(path)
    return paths


def build_manifest(raw_train_dir: Path | None = None) -> pd.DataFrame:
    raw_train_dir = raw_train_dir or RAW_TRAIN_DIR
    image_paths = discover_image_paths(raw_train_dir)

    if not image_paths:
        raise FileNotFoundError(
            f"No labeled images found under {raw_train_dir}. "
            "Add cat.*.jpg and dog.*.jpg files (flat or nested)."
        )

    rows: list[dict] = []
    dedup_rows: list[dict] = []

    for path in image_paths:
        label = parse_label(path.name)
        assert label is not None
        content_hash = hash_file(path)
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        rows.append(
            {
                "image_id": content_hash[:16],
                "filepath": rel_path,
                "filename": path.name,
                "label": label,
                "content_hash": content_hash,
                "group_id": content_hash,
                "split": "",
            }
        )

    manifest = pd.DataFrame(rows)

    # Dedup: keep first occurrence per content_hash; log duplicates.
    duplicated_hashes = manifest[manifest.duplicated("content_hash", keep=False)]
    if not duplicated_hashes.empty:
        for content_hash, group in duplicated_hashes.groupby("content_hash"):
            canonical = group.iloc[0]["filepath"]
            for _, row in group.iloc[1:].iterrows():
                dedup_rows.append(
                    {
                        "content_hash": content_hash,
                        "canonical_filepath": canonical,
                        "duplicate_filepath": row["filepath"],
                    }
                )
        manifest = manifest.drop_duplicates("content_hash", keep="first").reset_index(drop=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    dedup_df = pd.DataFrame(dedup_rows)
    dedup_df.to_csv(DEDUP_REPORT_PATH, index=False)

    nested = raw_train_dir / "train"
    if nested.is_dir() and any(nested.iterdir()):
        print(
            "Note: images were found in a nested train/ folder. "
            "Split will still work; consider flattening to data/raw/asirra/train/ later."
        )

    print(f"Wrote {len(manifest)} rows to {MANIFEST_PATH}")
    if not dedup_df.empty:
        print(f"Logged {len(dedup_df)} duplicate(s) in {DEDUP_REPORT_PATH}")

    metadata = {
        "files_scanned": len(image_paths),
        "unique_images": len(manifest),
        "duplicates_removed": len(dedup_df),
        "nested_train_folder": nested.is_dir() and any(nested.iterdir()),
    }
    return manifest, metadata


if __name__ == "__main__":
    build_manifest()
