"""Pipeline entry point for leak-free data prep and training stages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EARLY_STOPPING_PATIENCE, EPOCHS, MANIFEST_PATH, PROCESSED_TRAIN_DIR  # noqa: E402
from data.eda import run_eda  # noqa: E402
from data.manifest import build_manifest  # noqa: E402
from data.split import run_split  # noqa: E402
from models.train import train_baseline, train_optimize  # noqa: E402
from models.evaluate import TestEvaluationNotAllowed, evaluate_test, save_final_model  # noqa: E402
from visualizations import generate_pipeline_visuals, plot_split_counts_from_manifest  # noqa: E402
from reporting import (  # noqa: E402
    build_step01_manifest_report,
    build_step015_split_report,
    build_step02_eda_report,
    build_step03_train_baseline_report,
    build_step03_train_progress_report,
    build_step04_optimize_report,
    build_step05_save_report,
    refresh_reports_from_disk,
    run_leakage_tests,
    update_project_summary,
    write_step_report,
    write_test_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asirra cats vs dogs pipeline")
    parser.add_argument(
        "--stage",
        choices=("manifest", "split", "eda", "train", "visuals", "optimize", "evaluate", "save", "all", "report"),
        default="all",
        help=(
            "manifest: scan raw images; split: assign splits; eda: Step 2; "
            "train: Step 3 baseline CNN training; visuals: charts only (no retrain); "
            "optimize: Step 4 EarlyStopping + ModelCheckpoint; evaluate: one-time test (ALLOW_TEST_EVAL=true); "
            "save: Step 5 export final model + val reload check; "
            "report: regenerate summaries; all: manifest + split only"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs for --stage train (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for training generators",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Use literal VGG-from-scratch model (needs GPU/16GB+ RAM; default is VGG16 transfer)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=f"EarlyStopping patience for --stage optimize (default: {EARLY_STOPPING_PATIENCE})",
    )
    return parser.parse_args()


def _require_processed_splits() -> None:
    if not PROCESSED_TRAIN_DIR.exists() or not any(PROCESSED_TRAIN_DIR.rglob("*.jpg")):
        raise FileNotFoundError(
            "Processed train split not found. Run: python src/app.py --stage split"
        )


def stage_manifest() -> None:
    start = time.perf_counter()
    manifest, metadata = build_manifest()
    duration = time.perf_counter() - start
    report = build_step01_manifest_report(
        manifest,
        duration_seconds=duration,
        files_scanned=metadata["files_scanned"],
    )
    write_step_report("step01_manifest", report)
    update_project_summary()
    print(f"Report: reports/steps/step01_manifest.md")


def stage_split() -> None:
    start = time.perf_counter()
    manifest, _ = run_split(rebuild_manifest=not MANIFEST_PATH.exists())
    duration = time.perf_counter() - start
    plot_split_counts_from_manifest(manifest)
    report = build_step015_split_report(manifest, duration_seconds=duration)
    write_step_report("step015_split", report)
    test_results = run_leakage_tests()
    write_test_results(test_results)
    update_project_summary(test_results=test_results)
    print(f"Report: reports/steps/step015_split.md")
    print(f"Tests: reports/tests/test_results.md")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_eda() -> None:
    _require_processed_splits()
    start = time.perf_counter()
    metrics = run_eda()
    duration = time.perf_counter() - start
    report = build_step02_eda_report(metrics, duration_seconds=duration)
    write_step_report("step02_eda", report)

    test_results = None
    if (PROJECT_ROOT / "data/interim/splits.json").exists():
        test_results = run_leakage_tests()
        write_test_results(test_results)
    update_project_summary(test_results=test_results)

    print("EDA outputs:")
    for artifact in metrics["artifacts"]:
        print(f"  - {artifact}")
    print("Report: reports/steps/step02_eda.md")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_visuals() -> None:
    """Regenerate EDA/split/training-progress charts from disk — no model training."""
    _require_processed_splits()
    start = time.perf_counter()

    eda_metrics = run_eda()
    eda_report = build_step02_eda_report(eda_metrics, duration_seconds=time.perf_counter() - start)
    write_step_report("step02_eda", eda_report)

    manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else None
    visual_payload = generate_pipeline_visuals(manifest=manifest)

    if visual_payload.get("training_progress"):
        progress_report = build_step03_train_progress_report(visual_payload["training_progress"])
        write_step_report("step03_train_baseline", progress_report)

    update_project_summary()
    print("Visual aids (no retraining):")
    for artifact in visual_payload.get("artifacts", []):
        print(f"  - {artifact}")
    print("Report: reports/steps/step02_eda.md")
    if visual_payload.get("training_progress"):
        print("Report: reports/steps/step03_train_baseline.md (partial — from log)")


def stage_train(epochs: int | None = None, batch_size: int | None = None, from_scratch: bool = False) -> None:
    _require_processed_splits()
    use_transfer = not from_scratch
    if from_scratch:
        print("Warning: from-scratch VGG requires substantial RAM/GPU.")
    else:
        print(f"Using VGG16 transfer learning (default). Epochs: {epochs or 'config default'}")
    start = time.perf_counter()
    train_metrics = train_baseline(
        epochs=epochs,
        batch_size=batch_size,
        use_transfer=use_transfer,
    )
    duration = time.perf_counter() - start
    report = build_step03_train_baseline_report(train_metrics, duration_seconds=duration)
    write_step_report("step03_train_baseline", report)
    update_project_summary()
    print(f"Baseline val accuracy: {train_metrics['validation_summary'].get('final_val_accuracy')}")
    print(f"Best val accuracy: {train_metrics['validation_summary'].get('best_val_accuracy')}")
    print("Report: reports/steps/step03_train_baseline.md")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_optimize(epochs: int | None = None, batch_size: int | None = None, from_scratch: bool = False, patience: int | None = None) -> None:
    _require_processed_splits()
    use_transfer = not from_scratch
    print(
        f"Step 4 optimize — continuing from baseline with EarlyStopping "
        f"(patience={patience or EARLY_STOPPING_PATIENCE}), max epochs={epochs or EPOCHS}"
    )
    start = time.perf_counter()
    optimize_metrics = train_optimize(
        epochs=epochs,
        batch_size=batch_size,
        use_transfer=use_transfer,
        patience=patience,
    )
    duration = time.perf_counter() - start
    report = build_step04_optimize_report(optimize_metrics, duration_seconds=duration)
    write_step_report("step04_optimize", report)
    update_project_summary()
    print(f"Best val accuracy: {optimize_metrics['validation_summary'].get('best_val_accuracy')}")
    print(f"Early stopping triggered: {optimize_metrics.get('early_stopping_triggered')}")
    print("Report: reports/steps/step04_optimize.md")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_evaluate(batch_size: int | None = None, from_scratch: bool = False) -> None:
    _require_processed_splits()
    use_transfer = not from_scratch
    try:
        start = time.perf_counter()
        test_metrics = evaluate_test(batch_size=batch_size, use_transfer=use_transfer)
        duration = time.perf_counter() - start
    except TestEvaluationNotAllowed as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    from config import OPTIMIZED_VALIDATION_SUMMARY_PATH

    if OPTIMIZED_VALIDATION_SUMMARY_PATH.exists():
        with OPTIMIZED_VALIDATION_SUMMARY_PATH.open(encoding="utf-8") as handle:
            validation_summary = json.load(handle)
        optimize_metrics = {
            "validation_summary": validation_summary,
            "val_error_analysis": validation_summary.get("val_error_analysis", {}),
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
        report = build_step04_optimize_report(
            optimize_metrics,
            duration_seconds=duration,
            test_metrics=test_metrics,
        )
        write_step_report("step04_optimize", report)

    update_project_summary()
    print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test loss: {test_metrics['test_loss']:.4f}")
    print("Metrics: reports/metrics/final_test.json")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_save(batch_size: int | None = None, from_scratch: bool = False) -> None:
    _require_processed_splits()
    use_transfer = not from_scratch
    print("Step 5 save — exporting best checkpoint to asirra_cats_dogs_final.keras")
    start = time.perf_counter()
    save_metrics = save_final_model(batch_size=batch_size, use_transfer=use_transfer)
    duration = time.perf_counter() - start
    report = build_step05_save_report(save_metrics, duration_seconds=duration)
    write_step_report("step05_save", report)
    update_project_summary()

    reload = save_metrics["reload_sanity_check"]
    print(f"Exported: {save_metrics['final_model_path']}")
    print(f"Reload val accuracy: {reload['val_accuracy']:.4f}")
    print(f"Reload val loss: {reload['val_loss']:.4f}")
    print("Sanity check: reports/metrics/save_reload_sanity_check.json")
    print("Report: reports/steps/step05_save.md")
    print(f"Summary: reports/PROJECT_SUMMARY.md")


def stage_report() -> None:
    test_results = refresh_reports_from_disk()
    if test_results is None:
        print("No manifest found. Run: python src/app.py --stage all")
        return
    print(f"Summary: reports/PROJECT_SUMMARY.md")
    print(f"Tests: reports/tests/test_results.md")


def main() -> None:
    args = parse_args()

    if args.stage == "manifest":
        stage_manifest()
        return

    if args.stage == "split":
        stage_split()
        return

    if args.stage == "eda":
        stage_eda()
        return

    if args.stage == "visuals":
        stage_visuals()
        return

    if args.stage == "train":
        stage_train(epochs=args.epochs, batch_size=args.batch_size, from_scratch=args.from_scratch)
        return

    if args.stage == "optimize":
        stage_optimize(
            epochs=args.epochs,
            batch_size=args.batch_size,
            from_scratch=args.from_scratch,
            patience=args.patience,
        )
        return

    if args.stage == "evaluate":
        stage_evaluate(batch_size=args.batch_size, from_scratch=args.from_scratch)
        return

    if args.stage == "save":
        stage_save(batch_size=args.batch_size, from_scratch=args.from_scratch)
        return

    if args.stage == "report":
        stage_report()
        return

    stage_manifest()
    stage_split()


if __name__ == "__main__":
    main()
