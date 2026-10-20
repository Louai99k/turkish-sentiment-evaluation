"""Audit and aggregate one frozen five-seed final Transformer family.

CPU-only post-processing: no dataset download, model loading, or inference.
The complete and prespecified strict test views are recomputed from saved,
ordered predictions before any aggregate is written.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_metrics import classification_metrics
from src.protocol_predictions import (
    load_winvoker_test_lineage,
    normalize_label_ids,
    parse_boolean_series,
    stable_winvoker_sample_id,
)


ENDPOINTS = ("accuracy", "macro_f1", "weighted_f1", "negative_f1")
SUBSETS = ("historic_test_complete", "historic_test_strict")
SEEDS = (42, 52, 62, 72, 82)
DEFAULT_FROZEN_MATRIX = ROOT / "configs/final/multilingual_transformer_comparison_v1.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def verify_manifest(directory: Path, run_id: str) -> int:
    status = read_json(directory / "run_status.json")
    manifest = read_json(directory / "artifact_manifest.json")
    require_equal(status.get("status"), "complete", f"{run_id} status")
    require_equal(status.get("run_id"), run_id, f"{run_id} status ID")
    require_equal(manifest.get("run_id"), run_id, f"{run_id} manifest ID")
    require_equal(manifest.get("run_status"), "complete", f"{run_id} manifest status")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Empty/malformed artifact manifest: {run_id}")
    seen: set[str] = set()
    for item in files:
        relative = item["relative_to_run"]
        if relative in seen:
            raise ValueError(f"Duplicate manifest entry: {run_id}/{relative}")
        seen.add(relative)
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing/escaping manifest entry: {run_id}/{relative}")
        require_equal(path.stat().st_size, item["size_bytes"], f"{run_id}/{relative} size")
        require_equal(sha256_file(path), item["sha256"], f"{run_id}/{relative} SHA-256")
    for required in ("config.json", "metrics.json", "predictions.csv", "checkpoint_metadata.json", "checkpoints/best_checkpoint.pt"):
        if required not in seen:
            raise ValueError(f"Unmanifested required artifact: {run_id}/{required}")
    return len(files)


def validate_config(config: dict, matrix: dict, model: dict, seed: int, run_id: str) -> None:
    shared = matrix["shared_configuration"]
    pinned_model = read_json(ROOT / model["model_config"])
    for key, expected in {
        "run_id": run_id,
        "experiment_stage": "final",
        "evaluation_scope": "held_out_test",
        "model_family": model["model_family"],
        "model_config_sha256": model["model_config_sha256"],
        "model_id": pinned_model["model_id"],
        "model_revision": pinned_model["model_revision"],
        "tokenizer_id": pinned_model["tokenizer_id"],
        "tokenizer_revision": pinned_model["tokenizer_revision"],
        "pooling": model["pooling"],
        "seed": seed,
        "split_seed": 42,
        "epochs_maximum": shared["epochs"],
        "batch_size": shared["batch_size"],
        "max_length": shared["max_length"],
        "loss_mode": shared["loss_mode"],
        "dropout": shared["dropout"],
        "precision_mode": shared["precision_mode"],
        "local_files_only": shared["local_files_only"],
        "profile_id": "winvoker_canonical_v1",
    }.items():
        actual = config["preprocessing"].get(key) if key == "profile_id" else config.get(key)
        require_equal(actual, expected, f"{run_id} config {key}")
    for key in ("learning_rate",):
        if not math.isclose(float(config[key]), float(shared[key]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{run_id} config {key} differs from frozen matrix")
    require_equal(config["early_stopping"]["patience"], shared["early_stopping_patience"], f"{run_id} patience")
    require_equal(config["early_stopping"]["selection_metric"], "validation_weighted_f1", f"{run_id} selection")
    require_equal(config["preprocessing"]["profile_config_sha256"], matrix["preprocessing_profile_sha256"], f"{run_id} profile hash")
    require_equal(config["preprocessing_freeze"]["sha256"], matrix["preprocessing_freeze_manifest_sha256"], f"{run_id} freeze hash")
    require_equal(config["resolved_device"], "cuda", f"{run_id} device")
    require_equal(config["epochs_completed"], 3, f"{run_id} epochs")
    require_equal(config["dataset_revision"], "4bd9103ad5f24d44db806d40c94c8b9ec116ad05", f"{run_id} dataset revision")


def validate_predictions(frame: pd.DataFrame, lineage: pd.DataFrame, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    require_equal(len(frame), len(lineage), f"{run_id} prediction count")
    for column in ("sample_id", "test_position", "source_row_index", "source", "strict_included", "strict_unique_included", "canonical_cleaned_text_sha256", "cleaned_text_sha256", "preprocessing_profile_id", "true_id", "predicted_id"):
        if column not in frame:
            raise ValueError(f"{run_id} lacks prediction column {column}")
    if not frame["sample_id"].is_unique:
        raise ValueError(f"{run_id} contains duplicate sample IDs")
    expected_sample_ids = lineage["source_row_index"].map(stable_winvoker_sample_id).to_numpy()
    if not np.array_equal(frame["sample_id"].astype(str).to_numpy(), expected_sample_ids):
        raise ValueError(f"{run_id} stable sample-ID lineage mismatch")
    for actual_key, expected_key in (("test_position", "test_position"), ("source_row_index", "source_row_index"), ("source", "dataset"), ("canonical_cleaned_text_sha256", "cleaned_text_sha256"), ("cleaned_text_sha256", "cleaned_text_sha256")):
        if not np.array_equal(frame[actual_key].to_numpy(), lineage[expected_key].to_numpy()):
            raise ValueError(f"{run_id} ordered lineage mismatch: {actual_key}")
    for key in ("strict_included", "strict_unique_included"):
        actual = parse_boolean_series(frame[key], f"{run_id}.{key}").to_numpy()
        if not np.array_equal(actual, lineage[key].to_numpy()):
            raise ValueError(f"{run_id} ordered lineage mismatch: {key}")
    if not frame["preprocessing_profile_id"].eq("winvoker_canonical_v1").all():
        raise ValueError(f"{run_id} prediction profile mismatch")
    true = normalize_label_ids(frame["true_id"], f"{run_id}.true_id")
    predicted = normalize_label_ids(frame["predicted_id"], f"{run_id}.predicted_id")
    if not np.array_equal(true, lineage["label_id"].to_numpy()):
        raise ValueError(f"{run_id} true-label mismatch")
    return true, predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_FROZEN_MATRIX)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    if not matrix_path.is_relative_to(ROOT.resolve()) or not matrix_path.is_file():
        raise ValueError("--matrix must name an existing file inside the repository")
    matrix = read_json(matrix_path)
    require_equal(matrix["status"], "frozen", "final matrix status")
    require_equal(matrix["shared_configuration"]["seeds"], list(SEEDS), "final matrix seeds")
    matching_models = [item for item in matrix["models"] if item["model_family"] == args.family]
    if len(matching_models) != 1:
        available = [item.get("model_family") for item in matrix.get("models", [])]
        raise ValueError(
            f"--family {args.family!r} must identify exactly one matrix model; available={available}"
        )
    model = matching_models[0]
    frozen_files = [
        ("runner", matrix["runner"], matrix["runner_sha256"]),
        ("preprocessing profile", matrix["preprocessing_profile"], matrix["preprocessing_profile_sha256"]),
        ("preprocessing freeze", matrix["preprocessing_freeze_manifest"], matrix["preprocessing_freeze_manifest_sha256"]),
        ("model config", model["model_config"], model["model_config_sha256"]),
        ("runtime preflight", model["runtime_preflight_manifest"], model["runtime_preflight_manifest_sha256"]),
    ]
    for label, path_key, hash_key in (
        ("model factory", "model_factory", "model_factory_sha256"),
        ("trainer", "trainer", "trainer_sha256"),
        ("dataset config", "dataset_config", "dataset_config_sha256"),
        (
            "preprocessing fingerprints",
            "preprocessing_fingerprints",
            "preprocessing_fingerprints_sha256",
        ),
    ):
        if (path_key in matrix) != (hash_key in matrix):
            raise ValueError(f"Frozen matrix has incomplete {label} identity")
        if path_key in matrix:
            frozen_files.append((label, matrix[path_key], matrix[hash_key]))
    for label, path, expected in frozen_files:
        require_equal(sha256_file(ROOT / path), expected, f"frozen {label} SHA-256")
    require_equal(len(model["experiment_ids"]), len(SEEDS), "model run count")
    lineage = load_winvoker_test_lineage()
    strict = lineage["strict_included"].to_numpy(dtype=bool)
    output_dir = ROOT / "outputs/statistics" / args.run_id
    config = {
        "schema_version": "final-transformer-five-seed-aggregation-v1.0",
        "run_id": args.run_id,
        "family": args.family,
        "seeds": list(SEEDS),
        "subsets": list(SUBSETS),
        "endpoints": list(ENDPOINTS),
        "matrix_path": matrix_path.relative_to(ROOT).as_posix(),
        "matrix_sha256": sha256_file(matrix_path),
        "input_experiment_ids": model["experiment_ids"],
        "lineage_sha256": sha256_file(ROOT / "outputs/audits/historic_split_lineage_v1/test_inclusion_mask.csv"),
    }
    with ExperimentRun(output_dir, run_id=args.run_id, run_kind="final_transformer_five_seed_aggregation", config=config, runtime_environment={"execution_kind": "cpu_only_postprocessing"}) as run:
        inventory: list[dict] = []
        per_seed: list[dict] = []
        for seed, run_id in zip(SEEDS, model["experiment_ids"]):
            directory = ROOT / "outputs/experiments" / run_id
            count = verify_manifest(directory, run_id)
            source_config = read_json(directory / "config.json")
            validate_config(source_config, matrix, model, seed, run_id)
            checkpoint = read_json(directory / "checkpoint_metadata.json")
            checkpoint_metadata = checkpoint["metadata"]
            for key, expected in {
                "experiment_id": run_id,
                "model_family": model["model_family"],
                "model_revision": source_config["model_revision"],
                "tokenizer_id": source_config["tokenizer_id"],
                "tokenizer_revision": source_config["tokenizer_revision"],
                "model_config_sha256": model["model_config_sha256"],
                "training_seed": seed,
                "pooling": model["pooling"],
                "backbone_access": "src.transformer_model.TransformerSentimentClassifier.backbone",
                "posthoc_extraction_ready": True,
            }.items():
                require_equal(checkpoint_metadata.get(key), expected, f"{run_id} checkpoint {key}")
            require_equal(checkpoint["validation_weighted_f1"], source_config["best_validation_weighted_f1"], f"{run_id} selected F1")
            frame = pd.read_csv(directory / "predictions.csv", keep_default_na=False)
            true, predicted = validate_predictions(frame, lineage, run_id)
            stored = read_json(directory / "metrics.json")
            for subset_id, mask in ((SUBSETS[0], np.ones(len(lineage), dtype=bool)), (SUBSETS[1], strict)):
                computed = classification_metrics(true[mask], predicted[mask], subset_id=subset_id)
                reference = stored if subset_id == SUBSETS[0] else stored["views"][subset_id]
                require_equal(computed["n"], reference["n"], f"{run_id}/{subset_id} support")
                require_equal(computed["confusion_matrix"], reference["confusion_matrix"], f"{run_id}/{subset_id} confusion")
                for endpoint in ENDPOINTS:
                    if not math.isclose(computed[endpoint], reference[endpoint], rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"{run_id}/{subset_id}/{endpoint} stored/recomputed mismatch")
                    per_seed.append({"family": args.family, "seed": seed, "run_id": run_id, "subset_id": subset_id, "endpoint": endpoint, "value": computed[endpoint]})
            inventory.append({"family": args.family, "seed": seed, "run_id": run_id, "manifest_entries": count, "manifest_sha256": sha256_file(directory / "artifact_manifest.json"), "prediction_rows": len(frame), "selected_epoch": checkpoint["epoch_one_based"]})
            print(f"Accepted {run_id}: {count} manifested files, {len(frame):,} aligned predictions")
        aggregates: list[dict] = []
        for subset_id in SUBSETS:
            for endpoint in ENDPOINTS:
                values = [row["value"] for row in per_seed if row["subset_id"] == subset_id and row["endpoint"] == endpoint]
                aggregates.append({"family": args.family, "subset_id": subset_id, "endpoint": endpoint, "n_seeds": len(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)})
        lines = [f"# Final {args.family} five-seed aggregate", "", "Frozen seeds: `42, 52, 62, 72, 82`. The training seed is the uncertainty unit; all runs use the same held-out test split.", "", "| Subset | Endpoint | Mean | Sample SD |", "| :-- | :-- | --: | --: |"]
        lines += [f"| {row['subset_id']} | {row['endpoint']} | {row['mean']:.6f} | {row['sample_sd']:.6f} |" for row in aggregates]
        lines += ["", "Complete-test results preserve continuity; strict results are the prespecified exact-leakage sensitivity view.", "No model was loaded or trained by this aggregation.", ""]
        result = {"schema_version": config["schema_version"], "config": config, "input_inventory": inventory, "per_seed_metrics": per_seed, "aggregate_metrics": aggregates}
        run.write_csv("input_inventory.csv", inventory)
        run.write_csv("per_seed_metrics.csv", per_seed)
        run.write_csv("aggregate_metrics.csv", aggregates)
        run.write_json("results.json", result)
        run.write_text("report.md", "\n".join(lines))
        run.write_text("notes.md", "# Notes\n\nCPU-only audit and aggregation of the frozen five final Transformer seeds. No model loading, training, or inference.\n")
        run.declare_complete(("input_inventory.csv", "per_seed_metrics.csv", "aggregate_metrics.csv", "results.json", "report.md", "notes.md"))
    print(f"Run completed: {args.run_id}")


if __name__ == "__main__":
    main()
