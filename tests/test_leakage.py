"""Leakage prevention tests. See specs.md §10."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    MANIFEST_PATH,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    RAW_KAGGLE_TEST_DIR,
    SPLITS_PATH,
)
from models.evaluate import assert_test_eval_allowed  # noqa: E402,F401


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = MANIFEST_PATH
        cls.splits_path = SPLITS_PATH

    def test_manifest_exists_after_split(self) -> None:
        if not self.manifest_path.exists():
            self.skipTest("Run python src/app.py --stage all after adding raw images.")
        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        self.assertIn("split", manifest.columns)
        self.assertTrue(manifest["split"].isin(["train", "val", "test"]).all())

    def test_split_disjointness(self) -> None:
        if not self.manifest_path.exists():
            self.skipTest("Manifest not found.")

        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        image_ids = manifest.groupby("split")["image_id"].apply(set)
        train_ids = image_ids.get("train", set())
        val_ids = image_ids.get("val", set())
        test_ids = image_ids.get("test", set())

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_content_hash_not_shared_across_splits(self) -> None:
        if not self.manifest_path.exists():
            self.skipTest("Manifest not found.")

        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        for content_hash, group in manifest.groupby("content_hash"):
            splits = set(group["split"])
            self.assertEqual(len(splits), 1, f"Hash {content_hash} appears in multiple splits")

    def test_stratification_within_tolerance(self) -> None:
        if not self.manifest_path.exists():
            self.skipTest("Manifest not found.")

        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        overall_cat_ratio = (manifest["label"] == "cat").mean()

        for split in ("train", "val", "test"):
            subset = manifest[manifest["split"] == split]
            if subset.empty:
                continue
            split_ratio = (subset["label"] == "cat").mean()
            self.assertAlmostEqual(split_ratio, overall_cat_ratio, delta=0.02)

    def test_processed_paths_under_processed_dir(self) -> None:
        for split_dir in (PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR):
            self.assertIn("processed", str(split_dir))
            self.assertNotIn("raw", str(split_dir))

    def test_kaggle_test_dir_not_in_manifest(self) -> None:
        if not self.manifest_path.exists():
            self.skipTest("Manifest not found.")

        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        kaggle_prefix = str(RAW_KAGGLE_TEST_DIR.relative_to(PROJECT_ROOT))
        in_kaggle_test = manifest["filepath"].str.startswith(kaggle_prefix)
        self.assertFalse(in_kaggle_test.any())

    def test_splits_json_matches_manifest(self) -> None:
        if not self.manifest_path.exists() or not self.splits_path.exists():
            self.skipTest("Manifest or splits.json not found.")

        import pandas as pd

        manifest = pd.read_csv(self.manifest_path)
        with self.splits_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        for split in ("train", "val", "test"):
            expected = int((manifest["split"] == split).sum())
            self.assertEqual(summary["counts_by_split"][split], expected)

    def test_evaluate_gate_blocks_test_by_default(self) -> None:
        env = os.environ.copy()
        os.environ.pop("ALLOW_TEST_EVAL", None)
        try:
            import importlib

            import config
            import models.evaluate as evaluate_module

            importlib.reload(config)
            importlib.reload(evaluate_module)
            with self.assertRaises(evaluate_module.TestEvaluationNotAllowed):
                evaluate_module.assert_test_eval_allowed()
        finally:
            os.environ.clear()
            os.environ.update(env)


if __name__ == "__main__":
    unittest.main()
