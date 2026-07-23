"""Evaluation and prediction. See specs.md Step 4–5."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    ALLOW_TEST_EVAL,
    BASELINE_MODEL_PATH,
    FINAL_MODEL_PATH,
    FINAL_TEST_METRICS_PATH,
    METRICS_DIR,
    SAVED_MODELS_DIR,
    SAVED_MODEL_PATH,
    USE_TRANSFER_LEARNING,
)
from data.generators import build_generators


class TestEvaluationNotAllowed(RuntimeError):
    """Raised when held-out test evaluation is attempted without explicit opt-in."""


def assert_test_eval_allowed() -> None:
    if not ALLOW_TEST_EVAL:
        raise TestEvaluationNotAllowed(
            "Test set is locked until final evaluation. "
            "Set ALLOW_TEST_EVAL=true for a one-time test run."
        )


def _resolve_model_path(model_path=None):
    if model_path is not None:
        return model_path
    if SAVED_MODEL_PATH.exists():
        return SAVED_MODEL_PATH
    if BASELINE_MODEL_PATH.exists():
        return BASELINE_MODEL_PATH
    raise FileNotFoundError(
        "No trained model found. Run train and/or optimize stages first."
    )


def evaluate_test(
    *,
    model_path=None,
    batch_size: int | None = None,
    use_transfer: bool | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """One-time held-out test evaluation — gated by ALLOW_TEST_EVAL=true."""
    assert_test_eval_allowed()

    from tensorflow.keras.models import load_model

    resolved = _resolve_model_path(model_path)
    use_transfer = USE_TRANSFER_LEARNING if use_transfer is None else use_transfer
    model = load_model(resolved)
    generators = build_generators(
        batch_size=batch_size,
        use_vgg16_preprocess=use_transfer,
    )

    results = model.evaluate(
        generators.test,
        steps=generators.test_steps,
        verbose=verbose,
        return_dict=True,
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "allow_test_eval": True,
        "model_path": str(resolved),
        "test_samples": generators.test.samples,
        "test_steps": generators.test_steps,
        "test_loss": float(results.get("loss", 0.0)),
        "test_accuracy": float(results.get("accuracy", 0.0)),
        "metrics": {key: float(value) for key, value in results.items()},
        "note": "Single-run test evaluation. Do not re-run for tuning.",
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with FINAL_TEST_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def _file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def save_final_model(
    *,
    source_path: Path | str | None = None,
    batch_size: int | None = None,
    use_transfer: bool | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Export best checkpoint to final path and sanity-check reload on validation."""
    from tensorflow.keras.models import load_model

    resolved_source = Path(source_path) if source_path is not None else _resolve_model_path()
    if not resolved_source.exists():
        raise FileNotFoundError(f"Source model not found: {resolved_source}")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, FINAL_MODEL_PATH)

    use_transfer = USE_TRANSFER_LEARNING if use_transfer is None else use_transfer
    loaded = load_model(FINAL_MODEL_PATH)
    generators = build_generators(
        batch_size=batch_size,
        use_vgg16_preprocess=use_transfer,
    )

    results = loaded.evaluate(
        generators.val,
        steps=generators.validation_steps,
        verbose=verbose,
        return_dict=True,
    )

    model_paths = {
        "baseline_model": BASELINE_MODEL_PATH,
        "best_checkpoint": resolved_source,
        "final_export": FINAL_MODEL_PATH,
    }
    file_sizes_mb = {
        key: _file_size_mb(path) if path.exists() else None
        for key, path in model_paths.items()
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_model_path": str(resolved_source),
        "final_model_path": str(FINAL_MODEL_PATH),
        "export_method": "shutil.copy2 from best checkpoint",
        "file_sizes_mb": file_sizes_mb,
        "reload_sanity_check": {
            "split": "val",
            "val_samples": generators.val.samples,
            "validation_steps": generators.validation_steps,
            "val_loss": float(results.get("loss", 0.0)),
            "val_accuracy": float(results.get("accuracy", 0.0)),
            "metrics": {key: float(value) for key, value in results.items()},
        },
        "note": "Reload sanity check uses validation only — not the held-out test set.",
    }

    sanity_path = METRICS_DIR / "save_reload_sanity_check.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with sanity_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    payload["artifacts"] = [
        str(resolved_source),
        str(FINAL_MODEL_PATH),
        str(sanity_path),
    ]
    return payload
