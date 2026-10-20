"""Run one configuration-driven Transformer experiment under protocol v2.

The researcher launches this training command only after the required static,
preprocessing, and runtime gates. ``screening`` is structurally validation-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_berturk_experiment import (
    PRIMARY_METRIC_VIEW,
    REQUIRED_SCIENTIFIC_ARTIFACTS,
    collect_predictions,
    configure_reproducibility,
    resolve_device,
    runtime_environment,
)
from src.data_loader import (
    create_data_loaders,
    load_winvoker_dataset,
    preprocess_dataset_with_profile,
)
from src.evaluate import plot_confusion_matrix, plot_training_history
from src.experiment_artifacts import ExperimentRun, repo_relative, sha256_file
from src.preprocessing import (
    ORDERED_FINGERPRINT_ALGORITHM,
    fingerprint_preprocessed_splits,
    load_preprocessing_profile,
    load_profile_fingerprint_manifest,
    preprocessing_profile_record,
    verify_split_fingerprints,
)
from src.protocol_constants import (
    EXPERIMENT_SCHEMA_VERSION,
    ID_TO_LABEL,
    LABEL_IDS,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
    WINVOKER_SPLIT_SEED,
    WINVOKER_TEST_MASK_PATH,
    WINVOKER_TEST_MASK_SHA256,
)
from src.protocol_metrics import (
    build_evaluation_metrics,
    classification_metrics,
    confusion_matrix_rows,
    flatten_metric_views,
)
from src.protocol_predictions import (
    build_winvoker_prediction_frame,
    load_winvoker_test_lineage,
    sha256_text,
    stable_winvoker_sample_id,
)
from src.train import Trainer
from src.transformer_model import SUPPORTED_POOLING, get_transformer_model


DATASET_CONFIG_PATH = ROOT_DIR / "configs" / "datasets" / "winvoker_historic_v2.json"
MODEL_CONFIG_DIR = ROOT_DIR / "configs" / "models"
DEFAULT_PREPROCESSING_PROFILE = (
    ROOT_DIR / "configs" / "preprocessing" / "winvoker_canonical_v1.json"
)
DEFAULT_FINGERPRINT_MANIFEST = (
    ROOT_DIR
    / "outputs"
    / "audits"
    / "winvoker_preprocessing_profiles_v2"
    / "profile_fingerprints.json"
)
PREPROCESSING_FREEZE_SCHEMA_VERSION = "preprocessing-freeze-v1.0"
FINAL_FREEZE_REQUIRED_FAMILIES = {
    "berturk",
    "berturk_uncased",
    "mbert",
    "xlm_roberta",
    "turkish_electra",
    "distilberturk",
    "convberturk",
}
SCREENING_REQUIRED_ARTIFACTS = (
    "validation_metrics.json",
    "validation_predictions.csv",
    "validation_confusion_matrix.csv",
    "history.json",
    "checkpoint_metadata.json",
    "checkpoints/best_checkpoint.pt",
    "training_history.png",
    "validation_confusion_matrix.png",
    "notes.md",
)
SCREENING_FORBIDDEN_ARTIFACTS = (
    "metrics.json",
    "metric_views.csv",
    "confusion_matrices.csv",
    "predictions.csv",
    "confusion_matrix.png",
    "test_metrics.json",
    "test_predictions.csv",
    "test_confusion_matrix.csv",
    "test_confusion_matrix.png",
)
REQUIRED_MODEL_CONFIG_FIELDS = {
    "schema_version",
    "model_family",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "license",
    "architecture_family",
    "maximum_position_embeddings",
    "base_pretrained_checkpoint",
    "project_head",
    "official_model_page",
    "official_commit_page",
    "original_paper",
    "verified_on",
}


def is_full_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def load_model_config(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path if path.is_absolute() else ROOT_DIR / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(MODEL_CONFIG_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Model config must be inside {repo_relative(MODEL_CONFIG_DIR)}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Model config not found: {resolved}")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_MODEL_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"Model config is missing required fields: {missing}")
    if config["schema_version"] != "model-config-v2.0":
        raise ValueError(f"Unsupported model-config schema: {config['schema_version']!r}")
    for field in ("model_revision", "tokenizer_revision"):
        if not is_full_commit(config[field]):
            raise ValueError(f"{field} must be a full lowercase 40-character commit SHA")
    head = config["project_head"]
    if head.get("pooling") not in SUPPORTED_POOLING:
        raise ValueError(f"Unsupported pooling strategy: {head.get('pooling')!r}")
    if head.get("num_classes") != 3:
        raise ValueError("Protocol v2 requires a three-class project head")
    if head.get("classifier") == "linear":
        if "hidden_size" in head or "activation" in head or "dropout_placement" in head:
            raise ValueError("Linear head must retain its historical shape and placement")
    elif head.get("classifier") == "mlp":
        if (
            head.get("hidden_size") not in (128, 256)
            or head.get("activation") != "gelu"
            or head.get("dropout_placement") != "after_activation"
        ):
            raise ValueError("MLP head requires width 128/256, GELU, and post-activation dropout")
    else:
        raise ValueError(f"Unsupported classifier head: {head.get('classifier')!r}")
    if not config["base_pretrained_checkpoint"]:
        raise ValueError("Common benchmark configs must identify base pretrained checkpoints")
    return resolved, config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--preprocessing-profile", type=Path, default=DEFAULT_PREPROCESSING_PROFILE
    )
    parser.add_argument(
        "--preprocessing-fingerprints", type=Path, default=DEFAULT_FINGERPRINT_MANIFEST
    )
    parser.add_argument("--preprocessing-freeze-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-stage",
        choices=["diagnostic", "screening", "candidate", "final"],
        default="diagnostic",
    )
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--loss-mode", choices=["weighted", "unweighted", "focal"], default="unweighted"
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--focal-alpha", choices=["none", "balanced"], default="none")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("epochs, batch size, and max length must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.loss_mode == "focal":
        if args.focal_gamma is None or not np.isfinite(args.focal_gamma) or args.focal_gamma <= 0:
            parser.error("focal loss requires a finite --focal-gamma greater than zero")
        if args.exp_id is None:
            parser.error("focal loss requires an explicit --exp-id to distinguish gamma/alpha arms")
    elif args.focal_gamma is not None or args.focal_alpha != "none":
        parser.error("--focal-gamma and --focal-alpha apply only to --loss-mode focal")
    if args.exp_id is not None and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.exp_id)
        or ".." in args.exp_id
    ):
        parser.error("--exp-id must be a path-safe identifier without '..'")
    return args


def included_splits_for_stage(experiment_stage: str) -> tuple[str, ...]:
    """Return the only split frames the runner is allowed to construct."""

    return ("train", "val") if experiment_stage == "screening" else ("train", "val", "test")


def required_artifacts_for_stage(experiment_stage: str) -> tuple[str, ...]:
    return (
        SCREENING_REQUIRED_ARTIFACTS
        if experiment_stage == "screening"
        else REQUIRED_SCIENTIFIC_ARTIFACTS
    )


def assert_no_screening_test_artifacts(run: ExperimentRun) -> None:
    """Fail closed if a screening directory contains held-out-test outputs."""

    present = [
        relative_path
        for relative_path in SCREENING_FORBIDDEN_ARTIFACTS
        if run.path(relative_path).exists()
    ]
    if present:
        raise RuntimeError(
            "Validation-only screening directory contains forbidden held-out-test "
            f"artifacts: {present}"
        )


def default_experiment_id(
    model_family: str,
    profile_id: str,
    experiment_stage: str,
    seed: int,
    loss_mode: str,
    max_length: int,
) -> str:
    prefix = "screen" if experiment_stage == "screening" else "s1"
    return (
        f"{prefix}_{model_family}_winvoker_{profile_id}_len{max_length}_"
        f"seed{seed}_{loss_mode}_valf1_v2"
    )


def final_preprocessing_gate_required(experiment_stage: str, model_family: str) -> bool:
    return experiment_stage == "final" and model_family in FINAL_FREEZE_REQUIRED_FAMILIES


def validate_preprocessing_freeze(
    freeze: Mapping[str, Any],
    profile_record: Mapping[str, Any],
    fingerprint_manifest_sha256: str,
) -> None:
    expected = {
        "schema_version": PREPROCESSING_FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "profile_id": profile_record["profile_id"],
        "profile_config_sha256": profile_record["profile_config_sha256"],
        "profile_definition_sha256": profile_record["profile_definition_sha256"],
        "fingerprint_manifest_sha256": fingerprint_manifest_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": freeze.get(key)}
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    evidence = freeze.get("selection_evidence")
    if not isinstance(evidence, list) or not evidence:
        mismatches["selection_evidence"] = {
            "expected": "non-empty validation-only audit/screening evidence list",
            "observed": evidence,
        }
    if mismatches:
        raise ValueError(f"Preprocessing freeze manifest mismatch: {mismatches}")


def load_preprocessing_freeze(
    path: Path,
    profile_record: Mapping[str, Any],
    fingerprint_manifest_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    resolved = path if path.is_absolute() else ROOT_DIR / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Preprocessing freeze manifest not found: {resolved}")
    freeze = json.loads(resolved.read_text(encoding="utf-8"))
    validate_preprocessing_freeze(freeze, profile_record, fingerprint_manifest_sha256)
    return resolved, sha256_file(resolved), freeze


def resolved_config(
    args: argparse.Namespace,
    exp_id: str,
    model_config_path: Path,
    model_config: dict[str, Any],
    profile_record: dict[str, Any],
    fingerprint_manifest_path: Path,
    fingerprint_manifest_sha256: str,
    fingerprint_entry: dict[str, Any],
    freeze_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    screening = args.experiment_stage == "screening"
    config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": exp_id,
        "run_kind": (
            "transformer_validation_screening"
            if screening
            else "transformer_training_evaluation"
        ),
        "experiment_stage": args.experiment_stage,
        "evaluation_scope": "validation_only" if screening else "held_out_test",
        "held_out_test_access": {
            "split_frame_constructed": not screening,
            "loader_constructed": not screening,
            "lineage_mask_loaded": not screening,
            "predictions_generated": not screening,
            "metrics_generated": not screening,
            "absence_intentional": screening,
        },
        "model_family": model_config["model_family"],
        "model_id": model_config["model_id"],
        "model_revision": model_config["model_revision"],
        "tokenizer_id": model_config["tokenizer_id"],
        "tokenizer_revision": model_config["tokenizer_revision"],
        "model_config": repo_relative(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "architecture_family": model_config["architecture_family"],
        "pooling": model_config["project_head"]["pooling"],
        "head_configuration": {
            "classifier": model_config["project_head"]["classifier"],
            "hidden_size": model_config["project_head"].get("hidden_size"),
            "activation": model_config["project_head"].get("activation"),
            "dropout_placement": model_config["project_head"].get(
                "dropout_placement", "before_classifier"
            ),
        },
        "dataset_id": WINVOKER_DATASET_ID,
        "dataset_revision": WINVOKER_DATASET_REVISION,
        "dataset_config": repo_relative(DATASET_CONFIG_PATH),
        "dataset_config_sha256": sha256_file(DATASET_CONFIG_PATH),
        "upstream_split": "train",
        "split_seed": WINVOKER_SPLIT_SEED,
        "split_method": "locked ordered assignments from historic_split_lineage_v1",
        "materialized_splits": (
            ["train", "validation"]
            if screening
            else ["train", "validation", "test"]
        ),
        "test_split_id": "winvoker-historic-split42-test",
        "test_mask": repo_relative(WINVOKER_TEST_MASK_PATH),
        "test_mask_sha256": WINVOKER_TEST_MASK_SHA256,
        "preprocessing": {
            **profile_record,
            "fingerprint_manifest": repo_relative(fingerprint_manifest_path),
            "fingerprint_manifest_sha256": fingerprint_manifest_sha256,
            "ordered_fingerprint_algorithm": ORDERED_FINGERPRINT_ALGORITHM,
            "ordered_split_fingerprints": fingerprint_entry["splits"],
            "test_fingerprint_origin": (
                "precomputed_non_model_audit_not_accessed_by_screening_run"
                if screening
                else "precomputed_audit_and_verified_during_run"
            ),
        },
        "preprocessing_freeze": freeze_record,
        "label_order": [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)],
        "seed": args.seed,
        "epochs_maximum": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "loss_mode": args.loss_mode,
        "optimizer": {"name": "AdamW", "epsilon": 1e-8, "weight_decay": 0.01},
        "scheduler": {"name": "linear", "warmup_steps": 0},
        "gradient_clip_max_norm": 1.0,
        "early_stopping": {
            "selection_metric": "validation_weighted_f1",
            "direction": "maximize",
            "patience": args.early_stopping_patience,
            "tie_policy": "retain_earliest_epoch",
        },
        "precision_mode": "float32",
        "primary_metric_view": None if screening else PRIMARY_METRIC_VIEW,
        "device_request": args.device,
        "local_files_only": args.local_files_only,
        "resume_requested": args.resume,
    }
    if args.loss_mode == "focal":
        config["focal_loss"] = {
            "gamma": args.focal_gamma,
            "alpha_mode": args.focal_alpha,
            "alpha_source": (
                "balanced training-label class weights"
                if args.focal_alpha == "balanced"
                else "unit per-class weights"
            ),
            "formula": "mean(-alpha[y] * (1 - softmax(logits)[y])**gamma * log_softmax(logits)[y])",
            "reduction": "arithmetic_mean_over_examples",
            "focal_implementation_sha256": sha256_file(ROOT_DIR / "src" / "focal_loss.py"),
            "trainer_source_sha256": sha256_file(ROOT_DIR / "src" / "train.py"),
            "runner_source_sha256": sha256_file(Path(__file__)),
        }
    return config


def build_validation_prediction_frame(
    validation_frame: pd.DataFrame,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    logits: np.ndarray,
    profile_id: str,
) -> pd.DataFrame:
    expected_true = validation_frame["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_true, true_labels):
        raise ValueError("Validation true-label order differs from the locked validation split")
    source_rows = validation_frame["source_row_index"].to_numpy(dtype=np.int64)
    texts = validation_frame["text"].astype(str).reset_index(drop=True)
    result = pd.DataFrame(
        {
            "prediction_schema_version": "validation-predictions-v1.0",
            "sample_id": [stable_winvoker_sample_id(value) for value in source_rows],
            "dataset_id": WINVOKER_DATASET_ID,
            "dataset_revision": WINVOKER_DATASET_REVISION,
            "benchmark_split_id": "winvoker-historic-split42-validation",
            "validation_position": np.arange(len(validation_frame), dtype=np.int64),
            "source_row_index": source_rows,
            "source": validation_frame["dataset"].astype(str).to_numpy(),
            "preprocessing_profile_id": profile_id,
            "cleaned_text_sha256": [sha256_text(value) for value in texts],
            "text": texts,
            "true_id": true_labels,
            "predicted_id": predicted_labels,
            "true_label": [ID_TO_LABEL[value] for value in true_labels],
            "predicted_label": [ID_TO_LABEL[value] for value in predicted_labels],
            "prediction_correct": true_labels == predicted_labels,
        }
    )
    for position, label_id in enumerate(LABEL_IDS):
        result[f"logit_{ID_TO_LABEL[label_id].lower()}"] = logits[:, position]
    if not result["sample_id"].is_unique or not np.isfinite(logits).all():
        raise ValueError("Validation predictions have duplicate IDs or non-finite logits")
    return result


def build_profiled_test_prediction_frame(
    cleaned_texts,
    true_labels,
    predicted_labels,
    logits,
    profile_id: str,
) -> pd.DataFrame:
    """Retain canonical strict masks while hashing the selected profile's text."""

    lineage = load_winvoker_test_lineage()
    profiled_lineage = lineage.copy()
    canonical_hashes = profiled_lineage["cleaned_text_sha256"].astype(str).to_numpy()
    actual_hashes = pd.Series(cleaned_texts).astype(str).map(sha256_text).to_numpy()
    profiled_lineage["cleaned_text_sha256"] = actual_hashes
    predictions = build_winvoker_prediction_frame(
        profiled_lineage,
        cleaned_texts,
        true_labels,
        predicted_labels,
        scores=logits,
        score_kind="logit",
    )
    predictions.insert(
        predictions.columns.get_loc("cleaned_text_sha256"),
        "canonical_cleaned_text_sha256",
        canonical_hashes,
    )
    predictions.insert(
        predictions.columns.get_loc("cleaned_text_sha256") + 1,
        "preprocessing_profile_id",
        profile_id,
    )
    return predictions


def validation_confusion_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = metrics["confusion_matrix"]
    return [
        {
            "subset_id": metrics["subset_id"],
            "true_id": true_id,
            "true_label": ID_TO_LABEL[true_id],
            "predicted_id": predicted_id,
            "predicted_label": ID_TO_LABEL[predicted_id],
            "count": int(matrix[true_id][predicted_id]),
        }
        for true_id in LABEL_IDS
        for predicted_id in LABEL_IDS
    ]


def run_validation_only_screening(
    run: ExperimentRun,
    model,
    data_splits,
    data_config,
    device,
    config,
) -> float:
    """Evaluate only the locked validation loader; this function has no test input."""

    started = time.perf_counter()
    y_true, y_pred, logits = collect_predictions(model, data_config["loaders"]["val"], device)
    duration = time.perf_counter() - started
    predictions = build_validation_prediction_frame(
        data_splits["val"],
        y_true,
        y_pred,
        logits,
        config["preprocessing"]["profile_id"],
    )
    metrics = classification_metrics(
        y_true,
        y_pred,
        subset_id="historic_validation_complete",
        interpretation="validation_only_configuration_screening",
    )
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        label_mapping=data_config["label_mapping"],
        save_path=run.path("validation_confusion_matrix.png"),
    )
    run.write_json("validation_metrics.json", metrics)
    run.write_csv("validation_predictions.csv", predictions)
    run.write_csv("validation_confusion_matrix.csv", validation_confusion_rows(metrics))
    return duration


def run_held_out_test_evaluation(
    run: ExperimentRun,
    model,
    data_splits,
    data_config,
    device,
    config,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    y_true, y_pred, logits = collect_predictions(model, data_config["loaders"]["test"], device)
    duration = time.perf_counter() - started
    predictions = build_profiled_test_prediction_frame(
        data_splits["test"]["text"].to_numpy(),
        y_true,
        y_pred,
        logits,
        config["preprocessing"]["profile_id"],
    )
    metrics = build_evaluation_metrics(predictions, primary_view=PRIMARY_METRIC_VIEW)
    plot_confusion_matrix(
        np.asarray(metrics["views"][PRIMARY_METRIC_VIEW]["confusion_matrix"], dtype=np.int64),
        label_mapping=data_config["label_mapping"],
        save_path=run.path("confusion_matrix.png"),
    )
    run.write_json("metrics.json", metrics)
    run.write_csv("metric_views.csv", flatten_metric_views(metrics))
    run.write_csv("confusion_matrices.csv", confusion_matrix_rows(metrics))
    run.write_csv("predictions.csv", predictions)
    return duration, metrics


def experiment_notes(
    config: dict[str, Any],
    best_checkpoint: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> str:
    lines = [
        "# Experiment Notes",
        "",
        f"- Scientific stage: {config['experiment_stage']}",
        f"- Evaluation scope: `{config['evaluation_scope']}`",
        f"- Model: `{config['model_id']}@{config['model_revision']}`",
        f"- Tokenizer: `{config['tokenizer_id']}@{config['tokenizer_revision']}`",
        f"- Pooling/head: `{config['pooling']}` then `{config['head_configuration']}`; "
        f"dropout probability `{config['dropout']}`",
        f"- Preprocessing: `{config['preprocessing']['profile_id']}` "
        f"(`{config['preprocessing']['profile_config_sha256']}`)",
        f"- Loss mode: `{config['loss_mode']}`"
        + (f"; focal settings: `{config['focal_loss']}`" if "focal_loss" in config else ""),
        f"- Selected epoch (one-based): {int(best_checkpoint['epoch']) + 1}",
    ]
    if metrics is None:
        lines.extend(
            [
                "- Held-out test loader constructed: no",
                "- Held-out test frame/lineage mask loaded: no",
                "- Held-out test predictions/metrics written: no",
                "",
                "This screening artifact is validation-only and is not a paper result.",
            ]
        )
    else:
        primary = metrics["views"][PRIMARY_METRIC_VIEW]
        lines.extend(
            [
                f"- Complete-test Macro F1: {primary['macro_f1']:.6f}",
                f"- Complete-test Weighted F1: {primary['weighted_f1']:.6f}",
                f"- Complete-test Negative F1: {primary['negative_f1']:.6f}",
                "",
                "Complete, strict, polarity-only, and source views are retained.",
            ]
        )
    return "\n".join([*lines, ""])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_config_path, model_config = load_model_config(args.model_config)
    profile_path, profile = load_preprocessing_profile(args.preprocessing_profile)
    profile_record = preprocessing_profile_record(profile_path, profile)
    fingerprint_path, fingerprint_sha256, fingerprint_entry = load_profile_fingerprint_manifest(
        args.preprocessing_fingerprints,
        profile_record,
    )
    if args.max_length > int(model_config["maximum_position_embeddings"]):
        raise ValueError(
            f"--max-length {args.max_length} exceeds backbone limit "
            f"{model_config['maximum_position_embeddings']}"
        )

    freeze_record = None
    freeze_required = final_preprocessing_gate_required(
        args.experiment_stage, model_config["model_family"]
    )
    if freeze_required and args.preprocessing_freeze_manifest is None:
        raise ValueError(
            "Final Transformer training is blocked until --preprocessing-freeze-manifest "
            "names the validation-selected frozen profile"
        )
    if args.preprocessing_freeze_manifest is not None:
        freeze_path, freeze_sha256, freeze = load_preprocessing_freeze(
            args.preprocessing_freeze_manifest,
            profile_record,
            fingerprint_sha256,
        )
        freeze_record = {**freeze, "path": repo_relative(freeze_path), "sha256": freeze_sha256}

    reproducibility = configure_reproducibility(args.seed)
    device = resolve_device(args.device)
    exp_id = args.exp_id or default_experiment_id(
        model_config["model_family"],
        profile_record["profile_id"],
        args.experiment_stage,
        args.seed,
        args.loss_mode,
        args.max_length,
    )
    output_dir = ROOT_DIR / "outputs" / "experiments" / exp_id
    checkpoint_dir = output_dir / "checkpoints"
    resume_checkpoint = checkpoint_dir / "resume_checkpoint.pt"
    resume_from = resume_checkpoint if args.resume and resume_checkpoint.is_file() else None
    config = resolved_config(
        args,
        exp_id,
        model_config_path,
        model_config,
        profile_record,
        fingerprint_path,
        fingerprint_sha256,
        fingerprint_entry,
        freeze_record,
    )
    config["resume_checkpoint_found"] = resume_from is not None

    with ExperimentRun(
        output_dir,
        run_id=exp_id,
        run_kind=config["run_kind"],
        config=config,
        runtime_environment=runtime_environment(device, reproducibility),
        resume_incomplete=args.resume,
    ) as run:
        if args.experiment_stage == "screening":
            assert_no_screening_test_artifacts(run)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["tokenizer_id"],
            revision=model_config["tokenizer_revision"],
            local_files_only=args.local_files_only,
        )
        dataset = load_winvoker_dataset(
            dataset_name=WINVOKER_DATASET_ID,
            revision=WINVOKER_DATASET_REVISION,
        )
        included_splits = included_splits_for_stage(args.experiment_stage)
        data_splits = preprocess_dataset_with_profile(
            dataset,
            profile,
            included_splits=included_splits,
        )
        actual_fingerprints = fingerprint_preprocessed_splits(data_splits)
        verify_split_fingerprints(actual_fingerprints, fingerprint_entry["splits"])
        config["preprocessing"]["verified_during_run"] = actual_fingerprints
        data_config = create_data_loaders(
            data_splits,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )
        if args.experiment_stage == "screening" and "test" in data_config["loaders"]:
            raise RuntimeError("Validation-only screening unexpectedly constructed a test loader")

        model = get_transformer_model(
            model_config,
            dropout=args.dropout,
            device=device,
            local_files_only=args.local_files_only,
        )
        config["parameter_counts"] = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
        checkpoint_metadata = {
            "experiment_id": exp_id,
            "model_family": model_config["model_family"],
            "model_id": model_config["model_id"],
            "model_revision": model_config["model_revision"],
            "tokenizer_id": model_config["tokenizer_id"],
            "tokenizer_revision": model_config["tokenizer_revision"],
            "model_config_sha256": config["model_config_sha256"],
            "backbone_state_dict_prefix": "backbone.",
            "backbone_access": "src.transformer_model.TransformerSentimentClassifier.backbone",
            "pooling": model_config["project_head"]["pooling"],
            "preprocessing": config["preprocessing"],
            "stable_sample_lineage": {
                "dataset_id": WINVOKER_DATASET_ID,
                "dataset_revision": WINVOKER_DATASET_REVISION,
                "source_row_key": "source_row_index",
                "sample_id_function": "src.protocol_predictions.stable_winvoker_sample_id",
            },
            "posthoc_extraction_ready": True,
            "parameter_counts": config["parameter_counts"],
            "split_seed": WINVOKER_SPLIT_SEED,
            "training_seed": args.seed,
            "loss_mode": args.loss_mode,
            "epochs_maximum": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "dropout": args.dropout,
            "precision_mode": "float32",
            "early_stopping_patience": args.early_stopping_patience,
            "selection_metric": "validation_weighted_f1",
            "evaluation_scope": config["evaluation_scope"],
        }
        if model_config["project_head"]["classifier"] == "mlp":
            checkpoint_metadata["head_configuration"] = config["head_configuration"]
        if args.loss_mode == "focal":
            config["focal_loss"]["alpha_values_in_label_order"] = (
                [float(value) for value in data_config["class_weights"].cpu().numpy()]
                if args.focal_alpha == "balanced"
                else [1.0] * len(config["label_order"])
            )
            checkpoint_metadata["focal_loss"] = config["focal_loss"]
        trainer = Trainer(
            model=model,
            train_loader=data_config["loaders"]["train"],
            val_loader=data_config["loaders"]["val"],
            class_weights=data_config["class_weights"],
            device=device,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            output_dir=checkpoint_dir,
            loss_mode=args.loss_mode,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            history_path=run.path("history.json"),
            resume_from=resume_from,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_schema_version="transformer-checkpoint-v2.1",
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epochs_before = len(trainer.history["train_loss"])
        training_started = time.perf_counter()
        history = trainer.train(early_stopping_patience=args.early_stopping_patience)
        training_duration = time.perf_counter() - training_started
        epochs_this_invocation = len(history["train_loss"]) - epochs_before
        best_checkpoint = trainer.load_best_model()
        plot_training_history(history, save_path=run.path("training_history.png"))

        if args.experiment_stage == "screening":
            evaluation_duration = run_validation_only_screening(
                run, model, data_splits, data_config, device, config
            )
            metrics = None
        else:
            evaluation_duration, metrics = run_held_out_test_evaluation(
                run, model, data_splits, data_config, device, config
            )

        best_path = Path(trainer.best_model_path).resolve()
        evaluation_rows = (
            len(data_splits["val"]) if metrics is None else len(data_splits["test"])
        )
        config.update(
            {
                "resolved_device": str(device),
                "class_weights": [
                    float(value) for value in data_config["class_weights"].cpu().numpy()
                ],
                "epochs_completed": len(history["train_loss"]),
                "selected_epoch_zero_based": int(best_checkpoint["epoch"]),
                "selected_epoch_one_based": int(best_checkpoint["epoch"]) + 1,
                "best_validation_weighted_f1": float(best_checkpoint["val_f1"]),
                "best_validation_loss": float(best_checkpoint["val_loss"]),
                "best_checkpoint": repo_relative(best_path),
                "resume_checkpoint_retained_after_success": False,
                "efficiency": {
                    "training_duration_seconds": training_duration,
                    "evaluation_duration_seconds": evaluation_duration,
                    "evaluation_split": "validation" if metrics is None else "test",
                    "epochs_executed_this_invocation": epochs_this_invocation,
                    "training_examples_per_second": (
                        len(data_splits["train"]) * epochs_this_invocation
                    )
                    / training_duration,
                    "evaluation_examples_per_second": evaluation_rows / evaluation_duration,
                    "cuda_peak_memory_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                    "cuda_peak_memory_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else None
                    ),
                },
            }
        )
        checkpoint_record = {
            "schema_version": best_checkpoint["schema_version"],
            "checkpoint_role": best_checkpoint["checkpoint_role"],
            "path": repo_relative(best_path),
            "epoch_zero_based": int(best_checkpoint["epoch"]),
            "epoch_one_based": int(best_checkpoint["epoch"]) + 1,
            "selection_metric": best_checkpoint["selection_metric"],
            "validation_weighted_f1": float(best_checkpoint["val_f1"]),
            "validation_loss": float(best_checkpoint["val_loss"]),
            "metadata": best_checkpoint["metadata"],
        }
        run.write_json("config.json", config)
        run.write_json("history.json", history)
        run.write_json("checkpoint_metadata.json", checkpoint_record)
        run.write_text("notes.md", experiment_notes(config, best_checkpoint, metrics))
        trainer.clear_resume_checkpoint()
        if args.experiment_stage == "screening":
            assert_no_screening_test_artifacts(run)
        run.declare_complete(required_artifacts_for_stage(args.experiment_stage))


if __name__ == "__main__":
    main()
