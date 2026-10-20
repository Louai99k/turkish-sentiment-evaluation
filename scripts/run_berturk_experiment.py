"""Run one BERTurk experiment under the locked protocol-v2 contract.

This command performs training and evaluation. It is intended to be launched by
the researcher, normally inside the project Docker container. Every run writes a
transactional, checksummed artifact bundle under ``outputs/experiments``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import create_data_loaders, load_winvoker_dataset, preprocess_dataset
from src.evaluate import plot_confusion_matrix, plot_training_history
from src.experiment_artifacts import ExperimentRun, repo_relative
from src.model import get_model
from src.protocol_constants import (
    BERTURK_MODEL_ID,
    BERTURK_MODEL_REVISION,
    BERTURK_TOKENIZER_ID,
    BERTURK_TOKENIZER_REVISION,
    EXPERIMENT_SCHEMA_VERSION,
    ID_TO_LABEL,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
    WINVOKER_SPLIT_SEED,
    WINVOKER_TEST_MASK_PATH,
    WINVOKER_TEST_MASK_SHA256,
)
from src.protocol_metrics import (
    build_evaluation_metrics,
    confusion_matrix_rows,
    flatten_metric_views,
)
from src.protocol_predictions import (
    build_winvoker_prediction_frame,
    load_winvoker_test_lineage,
)
from src.train import Trainer


LEGACY_STAGE_MAP = {"dev": "diagnostic", "candidate": "candidate", "paper": "final"}
PRIMARY_METRIC_VIEW = "historic_test_complete"
MODEL_CONFIG_PATH = ROOT_DIR / "configs" / "models" / "berturk_base_cased_v1.json"
DATASET_CONFIG_PATH = ROOT_DIR / "configs" / "datasets" / "winvoker_historic_v2.json"
REQUIRED_SCIENTIFIC_ARTIFACTS = (
    "metrics.json",
    "metric_views.csv",
    "confusion_matrices.csv",
    "predictions.csv",
    "history.json",
    "checkpoint_metadata.json",
    "checkpoints/best_checkpoint.pt",
    "training_history.png",
    "confusion_matrix.png",
    "notes.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate BERTurk with protocol-v2 artifacts"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-stage",
        choices=["diagnostic", "screening", "candidate", "final"],
        default=None,
    )
    parser.add_argument(
        "--run-status",
        "--run_status",
        dest="legacy_run_status",
        choices=sorted(LEGACY_STAGE_MAP),
        default=None,
        help="Deprecated compatibility alias for the scientific experiment stage.",
    )
    parser.add_argument("--exp-id", "--exp_id", dest="exp_id", default=None)
    parser.add_argument("--model-name", "--model_name", default=BERTURK_MODEL_ID)
    parser.add_argument("--model-revision", default=BERTURK_MODEL_REVISION)
    parser.add_argument("--tokenizer-name", default=BERTURK_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=BERTURK_TOKENIZER_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=2e-5
    )
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=128)
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        dest="early_stopping_patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--loss-mode",
        "--loss_mode",
        dest="loss_mode",
        choices=["weighted", "unweighted"],
        default="weighted",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted/failed run; restart from epoch zero if no resume checkpoint exists.",
    )
    args = parser.parse_args(argv)

    legacy_stage = (
        LEGACY_STAGE_MAP[args.legacy_run_status] if args.legacy_run_status is not None else None
    )
    if args.experiment_stage and legacy_stage and args.experiment_stage != legacy_stage:
        parser.error("--experiment-stage conflicts with the legacy --run-status value")
    args.experiment_stage = args.experiment_stage or legacy_stage or "diagnostic"

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
    pinned_identifiers = {
        "--model-name": (args.model_name, BERTURK_MODEL_ID),
        "--model-revision": (args.model_revision, BERTURK_MODEL_REVISION),
        "--tokenizer-name": (args.tokenizer_name, BERTURK_TOKENIZER_ID),
        "--tokenizer-revision": (args.tokenizer_revision, BERTURK_TOKENIZER_REVISION),
    }
    for option, (actual, expected) in pinned_identifiers.items():
        if actual != expected:
            parser.error(
                f"{option} is locked by protocol v2: expected {expected!r}, got {actual!r}"
            )
    return args


def configure_reproducibility(seed: int) -> dict[str, Any]:
    python_hash_seed_at_startup = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "python_hash_seed_at_process_start": python_hash_seed_at_startup,
        "python_hash_seed_environment": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def runtime_environment(device: torch.device, reproducibility: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": str(device),
        "precision_mode": "float32",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "reproducibility": reproducibility,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        }
    else:
        payload["gpu"] = None
    return payload


def resolved_config(args: argparse.Namespace, exp_id: str) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": exp_id,
        "run_kind": "berturk_training_evaluation",
        "experiment_stage": args.experiment_stage,
        "model_family": "berturk",
        "model_id": args.model_name,
        "model_revision": args.model_revision,
        "tokenizer_id": args.tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "model_config": repo_relative(MODEL_CONFIG_PATH),
        "dataset_id": WINVOKER_DATASET_ID,
        "dataset_revision": WINVOKER_DATASET_REVISION,
        "dataset_config": repo_relative(DATASET_CONFIG_PATH),
        "upstream_split": "train",
        "split_seed": WINVOKER_SPLIT_SEED,
        "split_method": "stratified 80/20 followed by stratified 90/10 development split",
        "test_split_id": "winvoker-historic-split42-test",
        "test_mask": repo_relative(WINVOKER_TEST_MASK_PATH),
        "test_mask_sha256": WINVOKER_TEST_MASK_SHA256,
        "preprocessing": "src.data_loader.clean_text",
        "label_order": [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)],
        "seed": args.seed,
        "epochs_maximum": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "loss_mode": args.loss_mode,
        "optimizer": {"name": "AdamW", "epsilon": 1e-8},
        "scheduler": {"name": "linear", "warmup_steps": 0},
        "gradient_clip_max_norm": 1.0,
        "early_stopping": {
            "selection_metric": "validation_weighted_f1",
            "direction": "maximize",
            "patience": args.early_stopping_patience,
            "tie_policy": "retain_earliest_epoch",
        },
        "precision_mode": "float32",
        "primary_metric_view": PRIMARY_METRIC_VIEW,
        "device_request": args.device,
        "local_files_only": args.local_files_only,
        "resume_requested": args.resume,
        "legacy_run_status_argument": args.legacy_run_status,
    }


def default_experiment_id(seed: int, loss_mode: str) -> str:
    return f"s1_berturk_berturk_winvoker_seed{seed}_{loss_mode}_valf1_v2"


def collect_predictions(model, data_loader, device: torch.device) -> tuple[np.ndarray, ...]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            batch_logits = model(input_ids, attention_mask)
            labels.append(batch_labels.cpu().numpy())
            predictions.append(torch.argmax(batch_logits, dim=1).cpu().numpy())
            logits.append(batch_logits.cpu().numpy())
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(logits)


def experiment_notes(
    config: dict[str, Any], metrics: dict[str, Any], best_checkpoint: dict[str, Any]
) -> str:
    primary = metrics["views"][PRIMARY_METRIC_VIEW]
    return "\n".join(
        [
            "# Experiment Notes",
            "",
            f"- Scientific stage: {config['experiment_stage']}",
            "- Operational status: see `run_status.json`.",
            f"- Model: `{config['model_id']}@{config['model_revision']}`",
            f"- Tokenizer: `{config['tokenizer_id']}@{config['tokenizer_revision']}`",
            f"- Dataset: `{config['dataset_id']}@{config['dataset_revision']}`",
            f"- Training seed: {config['seed']}; locked split seed: {config['split_seed']}",
            f"- Loss: {config['loss_mode']} cross-entropy; precision: float32",
            f"- Selected epoch (one-based): {int(best_checkpoint['epoch']) + 1}",
            f"- Complete-test accuracy: {primary['accuracy']:.6f}",
            f"- Complete-test weighted F1: {primary['weighted_f1']:.6f}",
            f"- Complete-test macro F1: {primary['macro_f1']:.6f}",
            f"- Complete-test Negative F1: {primary['negative_f1']:.6f}",
            "",
            "## Interpretation boundary",
            "",
            "The complete historical test is the continuity headline. The strict, polarity-only, "
            "and source-specific views in `metrics.json` are required sensitivity diagnostics. "
            "Manuscript eligibility additionally depends on the prespecified multi-seed policy.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reproducibility = configure_reproducibility(args.seed)
    device = resolve_device(args.device)

    exp_id = args.exp_id or default_experiment_id(args.seed, args.loss_mode)
    output_dir = ROOT_DIR / "outputs" / "experiments" / exp_id
    checkpoint_dir = output_dir / "checkpoints"
    resume_checkpoint = checkpoint_dir / "resume_checkpoint.pt"
    resume_from = resume_checkpoint if args.resume and resume_checkpoint.is_file() else None

    config = resolved_config(args, exp_id)
    config["resume_checkpoint_found"] = resume_from is not None

    with ExperimentRun(
        output_dir,
        run_id=exp_id,
        run_kind="berturk_training_evaluation",
        config=config,
        runtime_environment=runtime_environment(device, reproducibility),
        resume_incomplete=args.resume,
    ) as run:
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if args.resume and resume_from is None:
            print("No epoch-boundary resume checkpoint exists; restarting training from epoch zero.")

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.tokenizer_revision,
            local_files_only=args.local_files_only,
        )
        dataset = load_winvoker_dataset(
            dataset_name=WINVOKER_DATASET_ID,
            revision=WINVOKER_DATASET_REVISION,
        )
        data_splits = preprocess_dataset(
            dataset,
            test_size=0.2,
            val_size=0.1,
            random_state=WINVOKER_SPLIT_SEED,
        )
        data_config = create_data_loaders(
            data_splits,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )

        model = get_model(
            num_classes=3,
            model_name=args.model_name,
            model_revision=args.model_revision,
            dropout=args.dropout,
            device=device,
            local_files_only=args.local_files_only,
        )
        checkpoint_metadata = {
            "experiment_id": exp_id,
            "model_id": args.model_name,
            "model_revision": args.model_revision,
            "tokenizer_id": args.tokenizer_name,
            "tokenizer_revision": args.tokenizer_revision,
            "dataset_id": WINVOKER_DATASET_ID,
            "dataset_revision": WINVOKER_DATASET_REVISION,
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
        }
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
            history_path=run.path("history.json"),
            resume_from=resume_from,
            checkpoint_metadata=checkpoint_metadata,
        )

        history = trainer.train(early_stopping_patience=args.early_stopping_patience)
        best_checkpoint = trainer.load_best_model()
        plot_training_history(history, save_path=run.path("training_history.png"))

        y_true, y_pred, logits = collect_predictions(
            model, data_config["loaders"]["test"], device
        )
        lineage = load_winvoker_test_lineage()
        predictions = build_winvoker_prediction_frame(
            lineage,
            data_splits["test"]["text"].to_numpy(),
            y_true,
            y_pred,
            scores=logits,
            score_kind="logit",
        )
        metrics = build_evaluation_metrics(predictions, primary_view=PRIMARY_METRIC_VIEW)
        primary_matrix = np.asarray(
            metrics["views"][PRIMARY_METRIC_VIEW]["confusion_matrix"], dtype=np.int64
        )
        plot_confusion_matrix(
            primary_matrix,
            label_mapping=data_config["label_mapping"],
            save_path=run.path("confusion_matrix.png"),
        )

        best_path = Path(trainer.best_model_path).resolve()
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
        run.write_json("metrics.json", metrics)
        run.write_csv("metric_views.csv", flatten_metric_views(metrics))
        run.write_csv("confusion_matrices.csv", confusion_matrix_rows(metrics))
        run.write_csv("predictions.csv", predictions)
        run.write_json("history.json", history)
        run.write_json("checkpoint_metadata.json", checkpoint_record)
        run.write_text("notes.md", experiment_notes(config, metrics, best_checkpoint))

        # Resume state is intentionally transient. Keep it on interruption/failure,
        # but remove it only after every final scientific artifact is present.
        trainer.clear_resume_checkpoint()
        run.declare_complete(REQUIRED_SCIENTIFIC_ARTIFACTS)
        primary = metrics["views"][PRIMARY_METRIC_VIEW]
        print(
            f"Complete-test metrics: accuracy={primary['accuracy']:.6f}, "
            f"weighted_f1={primary['weighted_f1']:.6f}, macro_f1={primary['macro_f1']:.6f}"
        )


if __name__ == "__main__":
    main()
