"""Run one locked, lineage-checked BiLSTM seed under protocol v2.

Seeds 52/62/72/82 are the original locked migration cohort. An explicit
seed-42 extension uses a new protocol-v2 run ID; the historical seed-42
output is preserved separately and is never modified by this command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_berturk_experiment import configure_reproducibility, resolve_device, runtime_environment
from scripts.run_stage2_neural_baselines import (
    BiLSTMSentimentClassifier,
    SequenceDataset,
    build_vocab,
    evaluate_torch_model,
    run_epoch,
    validate_epoch,
)
from src.data_loader import calculate_class_weights, load_winvoker_dataset, preprocess_dataset_with_profile
from src.evaluate import plot_confusion_matrix, plot_training_history
from src.experiment_artifacts import ExperimentRun, repo_relative, sha256_file
from src.preprocessing import (
    canonical_json_sha256,
    fingerprint_preprocessed_splits,
    load_preprocessing_profile,
    load_profile_fingerprint_manifest,
    preprocessing_profile_record,
    verify_split_fingerprints,
)
from src.protocol_constants import (
    EXPERIMENT_SCHEMA_VERSION,
    ID_TO_LABEL,
    WINVOKER_ASSIGNMENTS_PATH,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
    WINVOKER_SPLIT_SEED,
    WINVOKER_TEST_MASK_PATH,
    WINVOKER_TEST_MASK_SHA256,
)
from src.protocol_metrics import build_evaluation_metrics, confusion_matrix_rows, flatten_metric_views
from src.protocol_predictions import build_winvoker_prediction_frame, load_winvoker_test_lineage


LOCK_PATH = ROOT_DIR / "configs/final/bilstm_protocol_v2_v1.json"
LEGACY_RUNNER_PATH = ROOT_DIR / "scripts/run_stage2_neural_baselines.py"
RUNNER_PATH = Path(__file__).resolve()
REQUIRED = (
    "metrics.json", "metric_views.csv", "confusion_matrices.csv", "predictions.csv",
    "history.json", "checkpoint_metadata.json", "checkpoints/best_checkpoint.pt",
    "training_history.png", "confusion_matrix.png", "notes.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migrate-historical-seed42", action="store_true")
    args = parser.parse_args(argv)
    if args.seed == 42:
        if not args.migrate_historical_seed42:
            parser.error("Seed 42 requires --migrate-historical-seed42")
    elif args.migrate_historical_seed42:
        parser.error("--migrate-historical-seed42 is only valid with seed 42")
    elif args.seed not in (52, 62, 72, 82):
        parser.error("Only the four missing BiLSTM seeds 52/62/72/82 are eligible")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.exp_id) or ".." in args.exp_id:
        parser.error("--exp-id must be a path-safe identifier without '..'")
    expected = f"s2_neural_bilstm_winvoker_seed{args.seed}_protocol_v2"
    if args.exp_id != expected:
        parser.error(f"This locked run requires --exp-id {expected}")
    return args


def load_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "bilstm-run-config-v2.0",
        "model_family": "bilstm",
        "preprocessing_profile": "configs/preprocessing/winvoker_canonical_v1.json",
        "preprocessing_fingerprints": "outputs/audits/winvoker_preprocessing_profiles_v2/profile_fingerprints.json",
        "split_seed": 42,
        "training_seeds": [52, 62, 72, 82],
        "min_freq": 2,
        "max_vocab_size": 100000,
        "max_length": 128,
        "embedding_dim": 300,
        "hidden_dim": 128,
        "dropout": 0.3,
        "num_classes": 3,
        "epochs": 3,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "gradient_clip_max_norm": 1.0,
        "loss_mode": "balanced_weighted_cross_entropy",
        "selection_metric": "validation_weighted_f1",
        "tie_policy": "retain_earliest_epoch",
        "primary_metric_view": "historic_test_complete",
    }
    if lock != expected:
        raise ValueError("BiLSTM lock file differs from the reviewed continuity settings")
    if WINVOKER_SPLIT_SEED != lock["split_seed"]:
        raise ValueError("Historical split seed differs from the BiLSTM lock")
    return lock


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def selected_checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"checkpoints/selected_epoch{epoch + 1}.pt"


def make_loader(frame, vocab, lock, *, shuffle: bool, generator=None) -> DataLoader:
    dataset = SequenceDataset(
        frame["text"].to_numpy(), frame["label"].to_numpy(dtype=np.int64),
        vocab, lock["max_length"],
    )
    return DataLoader(
        dataset, batch_size=lock["batch_size"], shuffle=shuffle,
        num_workers=0, generator=generator,
    )


def write_notes(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    primary = metrics["views"]["historic_test_complete"]
    return "\n".join((
        "# BiLSTM protocol-v2 notes", "",
        f"- Run: `{config['run_id']}`",
        f"- Historical continuity lock: `{config['lock_config_sha256']}`",
        f"- Seed: {config['training_seed']}; split seed: 42",
        f"- Selected epoch: {config['selected_epoch_one_based']}",
        f"- Validation Weighted F1: {config['best_validation_weighted_f1']:.10f}",
        f"- Complete-test Weighted F1: {primary['weighted_f1']:.10f}",
        "- The historical seed-42 run remains separately preserved and conditional.",
        "- This run used locked train/validation/test assignments and canonical text.",
        "- Test loader and test predictions were created only after checkpoint selection.",
        "- This is a protocol-v2 migration, not a claim of bitwise identity with the historical runner.",
        "",
    ))


def run_seed(args: argparse.Namespace) -> None:
    lock = load_lock()
    profile_path, profile = load_preprocessing_profile(lock["preprocessing_profile"])
    profile_record = preprocessing_profile_record(profile_path, profile)
    if profile_record["profile_id"] != "winvoker_canonical_v1":
        raise ValueError("BiLSTM continuity requires the locked canonical profile")
    fingerprint_path, fingerprint_sha256, fingerprint_entry = load_profile_fingerprint_manifest(
        lock["preprocessing_fingerprints"], profile_record,
    )
    source_paths = {
        "runner": RUNNER_PATH,
        "historical_runner_reference": LEGACY_RUNNER_PATH,
        "reproducibility_helpers": ROOT_DIR / "scripts/run_berturk_experiment.py",
        "data_loader": ROOT_DIR / "src/data_loader.py",
        "evaluation_plots": ROOT_DIR / "src/evaluate.py",
        "experiment_artifacts": ROOT_DIR / "src/experiment_artifacts.py",
        "preprocessing": ROOT_DIR / "src/preprocessing.py",
        "protocol_constants": ROOT_DIR / "src/protocol_constants.py",
        "protocol_predictions": ROOT_DIR / "src/protocol_predictions.py",
        "protocol_metrics": ROOT_DIR / "src/protocol_metrics.py",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = {
        "run_id": args.exp_id,
        "seed": args.seed,
        "device_request": args.device,
        "lock_config_sha256": sha256_file(LOCK_PATH),
        "profile_config_sha256": profile_record["profile_config_sha256"],
        "fingerprint_manifest_sha256": fingerprint_sha256,
        "source_hashes": source_hashes,
    }
    output_dir = ROOT_DIR / "outputs/experiments" / args.exp_id
    best_path = output_dir / "checkpoints/best_checkpoint.pt"
    resume_path = output_dir / "checkpoints/resume_checkpoint.pt"
    if args.resume:
        status_path = output_dir / "run_status.json"
        if not status_path.is_file() or not resume_path.is_file():
            raise FileNotFoundError("Resume requires an incomplete run and epoch-boundary checkpoint")
        old_status = json.loads(status_path.read_text(encoding="utf-8"))
        if old_status.get("status") not in ("running", "failed", "interrupted"):
            raise ValueError("Only an incomplete run can be resumed")
        old_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if old_config.get("run_identity") != identity:
            raise ValueError("Resume identity differs from the saved run configuration")
    reproducibility = configure_reproducibility(args.seed)
    device = resolve_device(args.device)
    config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": args.exp_id,
        "run_kind": "bilstm_training_evaluation",
        "experiment_stage": "candidate",
        "evaluation_scope": "held_out_test_after_validation_selection",
        "model_family": "bilstm",
        "historical_comparability": "conditional_migration_not_bitwise_replay",
        "dataset_id": WINVOKER_DATASET_ID,
        "dataset_revision": WINVOKER_DATASET_REVISION,
        "split_seed": WINVOKER_SPLIT_SEED,
        "split_assignments": repo_relative(WINVOKER_ASSIGNMENTS_PATH),
        "split_assignments_sha256": WINVOKER_ASSIGNMENTS_SHA256,
        "test_mask": repo_relative(WINVOKER_TEST_MASK_PATH),
        "test_mask_sha256": WINVOKER_TEST_MASK_SHA256,
        "preprocessing": {
            **profile_record,
            "fingerprint_manifest": repo_relative(fingerprint_path),
            "fingerprint_manifest_sha256": fingerprint_sha256,
            "ordered_split_fingerprints": fingerprint_entry["splits"],
        },
        "lock_config": repo_relative(LOCK_PATH),
        "lock_config_sha256": identity["lock_config_sha256"],
        "source_hashes": source_hashes,
        "training_seed": args.seed,
        "device_request": args.device,
        "resume_requested": args.resume,
        "run_identity": identity,
        "training": lock,
        "primary_metric_view": lock["primary_metric_view"],
    }
    with ExperimentRun(
        output_dir, run_id=args.exp_id, run_kind=config["run_kind"], config=config,
        runtime_environment=runtime_environment(device, reproducibility),
        resume_incomplete=args.resume,
    ) as run:
        dataset = load_winvoker_dataset(
            dataset_name=WINVOKER_DATASET_ID, revision=WINVOKER_DATASET_REVISION,
        )
        train_val = preprocess_dataset_with_profile(
            dataset, profile, included_splits=("train", "val"),
        )
        verified = fingerprint_preprocessed_splits(train_val)
        verify_split_fingerprints(verified, fingerprint_entry["splits"])
        config["preprocessing"]["verified_before_training"] = verified
        vocab = build_vocab(
            train_val["train"]["text"].to_numpy(),
            min_freq=lock["min_freq"], max_vocab_size=lock["max_vocab_size"],
        )
        vocab_sha256 = canonical_json_sha256(vocab)
        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = make_loader(
            train_val["train"], vocab, lock, shuffle=True, generator=train_generator,
        )
        val_loader = make_loader(train_val["val"], vocab, lock, shuffle=False)
        model = BiLSTMSentimentClassifier(
            vocab_size=len(vocab), embedding_dim=lock["embedding_dim"],
            hidden_dim=lock["hidden_dim"], num_classes=lock["num_classes"],
            dropout=lock["dropout"],
        ).to(device)
        weights = calculate_class_weights(train_val["train"]["label"].to_numpy()).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(
            model.parameters(), lr=lock["learning_rate"],
            weight_decay=lock["weight_decay"], eps=lock["adam_epsilon"],
        )
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_f1 = -float("inf")
        selected_epoch = None
        start_epoch = 0
        if args.resume:
            resume = torch.load(resume_path, map_location=device, weights_only=True)
            if resume.get("run_identity") != identity or resume.get("vocab_sha256") != vocab_sha256:
                raise ValueError("Resume checkpoint identity or vocabulary mismatch")
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            history = resume["history"]
            best_f1 = float(resume["best_validation_weighted_f1"])
            selected_epoch = int(resume["selected_epoch_zero_based"])
            start_epoch = int(resume["completed_epochs"])
            if start_epoch < 1 or start_epoch > lock["epochs"] or any(
                len(values) != start_epoch for values in history.values()
            ) or not selected_checkpoint_path(output_dir, selected_epoch).is_file():
                raise ValueError("Resume epoch/history/best-checkpoint state is inconsistent")
            torch.set_rng_state(resume["torch_rng_state"].cpu())
            if device.type == "cuda":
                if resume.get("cuda_rng_states") is None:
                    raise ValueError("CUDA RNG state is absent from resume checkpoint")
                torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
            train_generator.set_state(resume["train_generator_state"].cpu())
            print(f"Resuming at epoch {start_epoch + 1}; completed epochs remain preserved")

        training_started = time.perf_counter()
        for epoch in range(start_epoch, lock["epochs"]):
            print(f"Epoch {epoch + 1}/{lock['epochs']}")
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_f1"].append(float(val_f1))
            print(f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_f1={val_f1:.10f}")
            if val_f1 > best_f1:
                best_f1 = float(val_f1)
                selected_epoch = epoch
                save_checkpoint(selected_checkpoint_path(output_dir, epoch), {
                    "schema_version": "bilstm-checkpoint-v2.0",
                    "checkpoint_role": "validation_selected_best",
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "val_loss": float(val_loss),
                    "history_at_selection": {key: list(value) for key, value in history.items()},
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "vocab_sha256": vocab_sha256,
                    "model_config": {
                        "embedding_dim": lock["embedding_dim"],
                        "hidden_dim": lock["hidden_dim"],
                        "dropout": lock["dropout"],
                        "max_length": lock["max_length"],
                        "num_classes": lock["num_classes"],
                    },
                    "run_identity": identity,
                    "preprocessing": config["preprocessing"],
                    "stable_sample_lineage": {
                        "source_row_key": "source_row_index",
                        "sample_id_function": "src.protocol_predictions.stable_winvoker_sample_id",
                    },
                    "pooling": "concatenated_final_bidirectional_lstm_hidden_states",
                    "embedding_access": "scripts.run_stage2_neural_baselines.BiLSTMSentimentClassifier.embedding",
                })
            save_checkpoint(resume_path, {
                "schema_version": "bilstm-resume-v2.0",
                "run_identity": identity,
                "vocab_sha256": vocab_sha256,
                "completed_epochs": epoch + 1,
                "selected_epoch_zero_based": selected_epoch,
                "best_validation_weighted_f1": best_f1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "train_generator_state": train_generator.get_state(),
            })
            run.write_json("history.json", history)
        training_duration = time.perf_counter() - training_started
        selected_path = selected_checkpoint_path(output_dir, selected_epoch)
        best = torch.load(selected_path, map_location=device, weights_only=True)
        if best.get("run_identity") != identity or best.get("vocab_sha256") != vocab_sha256:
            raise ValueError("Selected checkpoint identity or vocabulary mismatch")
        if int(best["epoch"]) != selected_epoch or float(best["val_f1"]) != best_f1:
            raise ValueError("Selected checkpoint differs from validation history")
        best_temp = best_path.with_name(best_path.name + ".tmp")
        shutil.copyfile(selected_path, best_temp)
        os.replace(best_temp, best_path)
        model.load_state_dict(best["model_state_dict"])
        plot_training_history(history, save_path=run.path("training_history.png"))

        # Held-out text and lineage are deliberately first materialized here.
        test_frame = preprocess_dataset_with_profile(
            dataset, profile, included_splits=("test",),
        )["test"]
        test_fingerprint = fingerprint_preprocessed_splits({"test": test_frame})
        verify_split_fingerprints(test_fingerprint, fingerprint_entry["splits"])
        lineage = load_winvoker_test_lineage()
        if not np.array_equal(
            test_frame["source_row_index"].to_numpy(dtype=np.int64),
            lineage["source_row_index"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Test source-row lineage differs from locked mask")
        test_loader = make_loader(test_frame, vocab, lock, shuffle=False)
        evaluation_started = time.perf_counter()
        y_true, y_pred, logits = evaluate_torch_model(model, test_loader, device)
        evaluation_duration = time.perf_counter() - evaluation_started
        predictions = build_winvoker_prediction_frame(
            lineage, test_frame["text"].to_numpy(), y_true, y_pred,
            scores=logits, score_kind="logit",
        )
        metrics = build_evaluation_metrics(predictions, primary_view=lock["primary_metric_view"])
        plot_confusion_matrix(
            np.asarray(metrics["views"][lock["primary_metric_view"]]["confusion_matrix"]),
            label_mapping=ID_TO_LABEL, save_path=run.path("confusion_matrix.png"),
        )
        config.update({
            "resolved_device": str(device),
            "vocab_size": len(vocab),
            "vocab_sha256": vocab_sha256,
            "class_weights": [float(value) for value in weights.cpu().numpy()],
            "epochs_completed": len(history["train_loss"]),
            "selected_epoch_zero_based": selected_epoch,
            "selected_epoch_one_based": selected_epoch + 1,
            "best_validation_weighted_f1": best_f1,
            "best_checkpoint": repo_relative(best_path),
            "preprocessing_verified_after_selection": test_fingerprint,
            "efficiency": {
                "training_duration_seconds_this_invocation": training_duration,
                "evaluation_duration_seconds": evaluation_duration,
            },
        })
        run.write_json("config.json", config)
        run.write_json("history.json", history)
        run.write_json("metrics.json", metrics)
        run.write_csv("predictions.csv", predictions)
        run.write_csv("metric_views.csv", flatten_metric_views(metrics))
        run.write_csv("confusion_matrices.csv", confusion_matrix_rows(metrics))
        run.write_json("checkpoint_metadata.json", {
            "schema_version": best["schema_version"],
            "path": repo_relative(best_path),
            "sha256": sha256_file(best_path),
            "selected_epoch_zero_based": selected_epoch,
            "validation_weighted_f1": best_f1,
            "vocab_sha256": vocab_sha256,
            "run_identity": identity,
            "pooling": best["pooling"],
            "stable_sample_lineage": best["stable_sample_lineage"],
        })
        run.write_text("notes.md", write_notes(config, metrics))
        resume_path.unlink(missing_ok=True)
        run.declare_complete(REQUIRED)


def main(argv: Sequence[str] | None = None) -> None:
    run_seed(parse_args(argv))


if __name__ == "__main__":
    main()
