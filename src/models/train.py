"""Training loop for baseline and optimized models. See specs.md Steps 3–4."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from config import (
    BASELINE_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    METRICS_DIR,
    OPTIMIZED_TRAINING_CURVES_PATH,
    OPTIMIZED_TRAINING_HISTORY_PATH,
    OPTIMIZED_VALIDATION_SUMMARY_PATH,
    SAVED_MODEL_PATH,
    SAVED_MODELS_DIR,
    TRAINING_HISTORY_PATH,
    TRANSFER_EPOCHS,
    USE_TRANSFER_LEARNING,
    VALIDATION_SUMMARY_PATH,
)
from data.generators import build_generators
from models.vgg import build_model, build_transfer_model
from visualizations import (
    _patience_counter,
    plot_early_stopping_summary,
    plot_training_curves,
    plot_val_error_analysis,
)


def _history_to_dict(history) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def _summarize_history(history_dict: dict[str, list[float]]) -> dict[str, Any]:
    epochs = len(history_dict.get("loss", []))
    summary: dict[str, Any] = {"epochs_completed": epochs, "per_epoch": []}

    for index in range(epochs):
        summary["per_epoch"].append(
            {
                "epoch": index + 1,
                "loss": history_dict.get("loss", [None])[index],
                "accuracy": history_dict.get("accuracy", [None])[index],
                "val_loss": history_dict.get("val_loss", [None])[index],
                "val_accuracy": history_dict.get("val_accuracy", [None])[index],
            }
        )

    if epochs:
        summary["final_train_loss"] = history_dict["loss"][-1]
        summary["final_train_accuracy"] = history_dict.get("accuracy", [None])[-1]
        summary["final_val_loss"] = history_dict.get("val_loss", [None])[-1]
        summary["final_val_accuracy"] = history_dict.get("val_accuracy", [None])[-1]
        summary["best_val_accuracy"] = max(history_dict.get("val_accuracy", [0.0]))

    return summary


def _write_json(path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_training_model(*, use_transfer: bool):
    if use_transfer:
        return build_transfer_model(), "vgg16_transfer"
    return build_model(), "vgg_style_from_scratch"


def train_baseline(
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    use_transfer: bool | None = None,
    callbacks: list[Callback] | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Step 3 training on full train split — validation on val only, no test data."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    use_transfer = USE_TRANSFER_LEARNING if use_transfer is None else use_transfer
    model, architecture = _build_training_model(use_transfer=use_transfer)
    generators = build_generators(
        batch_size=batch_size,
        use_vgg16_preprocess=use_transfer,
    )

    if epochs is not None:
        epoch_count = epochs
    elif use_transfer:
        epoch_count = TRANSFER_EPOCHS
    else:
        epoch_count = EPOCHS

    fit_callbacks = list(callbacks or [])
    fit_callbacks.append(
        ModelCheckpoint(
            filepath=str(BASELINE_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
    )

    history = model.fit(
        generators.train,
        steps_per_epoch=generators.steps_per_epoch,
        epochs=epoch_count,
        validation_data=generators.val,
        validation_steps=generators.validation_steps,
        callbacks=fit_callbacks,
        verbose=verbose,
    )

    history_dict = _history_to_dict(history)
    validation_summary = _summarize_history(history_dict)

    _write_json(TRAINING_HISTORY_PATH, history_dict)
    _write_json(VALIDATION_SUMMARY_PATH, validation_summary)
    model.save(BASELINE_MODEL_PATH)

    training_curves_path = plot_training_curves(history_dict)
    val_error_analysis = plot_val_error_analysis(model, generators.val)

    trainable_params = getattr(model, "_transfer_metadata", {}).get("trainable_params")
    frozen_params = getattr(model, "_transfer_metadata", {}).get("frozen_params")

    return {
        "architecture": architecture,
        "use_transfer_learning": use_transfer,
        "epochs": epoch_count,
        "batch_size": batch_size or generators.train.batch_size,
        "train_samples": generators.train.samples,
        "val_samples": generators.val.samples,
        "test_samples": generators.test.samples,
        "steps_per_epoch": generators.steps_per_epoch,
        "validation_steps": generators.validation_steps,
        "test_steps": generators.test_steps,
        "total_params": int(model.count_params()),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "num_layers": len(model.layers),
        "history": history_dict,
        "validation_summary": validation_summary,
        "val_error_analysis": val_error_analysis,
        "artifacts": [
            str(TRAINING_HISTORY_PATH),
            str(VALIDATION_SUMMARY_PATH),
            str(BASELINE_MODEL_PATH),
            training_curves_path,
            *val_error_analysis["artifacts"],
        ],
    }


def train_optimize(
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    use_transfer: bool | None = None,
    patience: int | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Step 4 — continue from baseline with EarlyStopping and ModelCheckpoint on val_accuracy."""
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {BASELINE_MODEL_PATH}. Run: python src/app.py --stage train"
        )

    from tensorflow.keras.models import load_model

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    use_transfer = USE_TRANSFER_LEARNING if use_transfer is None else use_transfer
    epoch_count = epochs if epochs is not None else EPOCHS
    stop_patience = patience if patience is not None else EARLY_STOPPING_PATIENCE

    model = load_model(BASELINE_MODEL_PATH)
    generators = build_generators(
        batch_size=batch_size,
        use_vgg16_preprocess=use_transfer,
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=stop_patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=str(SAVED_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        generators.train,
        steps_per_epoch=generators.steps_per_epoch,
        epochs=epoch_count,
        validation_data=generators.val,
        validation_steps=generators.validation_steps,
        callbacks=[early_stop, checkpoint],
        verbose=verbose,
    )

    history_dict = _history_to_dict(history)
    validation_summary = _summarize_history(history_dict)
    validation_summary["early_stopping_patience"] = stop_patience
    validation_summary["max_epochs_requested"] = epoch_count
    validation_summary["architecture"] = "vgg16_transfer" if use_transfer else "vgg_style_from_scratch"
    validation_summary["train_samples"] = generators.train.samples
    validation_summary["val_samples"] = generators.val.samples
    if history_dict.get("val_accuracy"):
        best_index = int(np.argmax(history_dict["val_accuracy"]))
        validation_summary["best_epoch"] = best_index + 1
        validation_summary["best_val_accuracy"] = history_dict["val_accuracy"][best_index]

    if VALIDATION_SUMMARY_PATH.exists():
        with VALIDATION_SUMMARY_PATH.open(encoding="utf-8") as handle:
            baseline_summary = json.load(handle)
        validation_summary["baseline_val_accuracy"] = baseline_summary.get("best_val_accuracy")

    optimize_vals = [
        float(row["val_accuracy"])
        for row in validation_summary.get("per_epoch", [])
        if row.get("val_accuracy") is not None
    ]
    waits = _patience_counter(optimize_vals)
    validation_summary["early_stopping_triggered"] = (
        bool(waits)
        and waits[-1] >= stop_patience
        and validation_summary["epochs_completed"] < epoch_count
    )
    validation_summary["manually_terminated"] = (
        validation_summary["epochs_completed"] < epoch_count
        and not validation_summary["early_stopping_triggered"]
    )
    if validation_summary["manually_terminated"]:
        validation_summary["terminated_during_epoch"] = validation_summary["epochs_completed"] + 1

    _write_json(OPTIMIZED_TRAINING_HISTORY_PATH, history_dict)
    _write_json(OPTIMIZED_VALIDATION_SUMMARY_PATH, validation_summary)

    if SAVED_MODEL_PATH.exists():
        model = load_model(SAVED_MODEL_PATH)
    else:
        model.save(SAVED_MODEL_PATH)

    training_curves_path = plot_training_curves(
        history_dict,
        output_path=OPTIMIZED_TRAINING_CURVES_PATH,
        title="Step 4 optimize training curves (validation on val split only)",
    )
    early_stopping_summary_path = plot_early_stopping_summary(validation_summary)
    val_error_analysis = plot_val_error_analysis(model, generators.val)

    architecture = "vgg16_transfer" if use_transfer else "vgg_style_from_scratch"
    trainable_params = getattr(model, "_transfer_metadata", {}).get("trainable_params")
    frozen_params = getattr(model, "_transfer_metadata", {}).get("frozen_params")

    return {
        "architecture": architecture,
        "use_transfer_learning": use_transfer,
        "continued_from": str(BASELINE_MODEL_PATH),
        "epochs_requested": epoch_count,
        "epochs_completed": len(history_dict.get("loss", [])),
        "early_stopping_triggered": validation_summary["early_stopping_triggered"],
        "early_stopping_patience": stop_patience,
        "batch_size": batch_size or generators.train.batch_size,
        "train_samples": generators.train.samples,
        "val_samples": generators.val.samples,
        "test_samples": generators.test.samples,
        "steps_per_epoch": generators.steps_per_epoch,
        "validation_steps": generators.validation_steps,
        "test_steps": generators.test_steps,
        "total_params": int(model.count_params()),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "num_layers": len(model.layers),
        "history": history_dict,
        "validation_summary": validation_summary,
        "val_error_analysis": val_error_analysis,
        "artifacts": [
            str(OPTIMIZED_TRAINING_HISTORY_PATH),
            str(OPTIMIZED_VALIDATION_SUMMARY_PATH),
            str(SAVED_MODEL_PATH),
            training_curves_path,
            *([early_stopping_summary_path] if early_stopping_summary_path else []),
            *val_error_analysis["artifacts"],
        ],
    }
