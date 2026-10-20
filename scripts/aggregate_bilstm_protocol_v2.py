"""Audit five new protocol-v2 BiLSTM runs without loading or training a model.

The preserved historical seed-42 result is continuity evidence only. The
primary five-seed cohort uses a separate new protocol-v2 seed-42 run.
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
from src.protocol_constants import ID_TO_LABEL, WINVOKER_DATASET_REVISION
from src.protocol_metrics import classification_metrics
from src.protocol_predictions import (
    load_winvoker_test_lineage,
    normalize_label_ids,
    parse_boolean_series,
    sha256_text,
    stable_winvoker_sample_id,
)


SEEDS = (42, 52, 62, 72, 82)
MODERN_SEEDS = (52, 62, 72, 82)
ENDPOINTS = ("accuracy", "macro_f1", "weighted_f1", "negative_f1")
SUBSETS = ("historic_test_complete", "historic_test_strict")
RUN_ID = "bilstm_five_seed_protocol_v2_v1"
HISTORICAL_ID = "s2_neural_bilstm_winvoker_seed42_base"
PREFLIGHT_FOUR = ROOT / "outputs/audits/bilstm_runner_v2_preflight_v1/results.json"
PREFLIGHT_SEED42 = ROOT / "outputs/audits/bilstm_runner_v2_seed42_extension_preflight_v1/results.json"
CHECKPOINT_AUDIT = ROOT / "outputs/audits/bilstm_seed42_migration_gate_v1/results.json"
LOCK = ROOT / "configs/final/bilstm_protocol_v2_v1.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_manifest(directory: Path, run_id: str) -> tuple[int, dict[str, str]]:
    status = read_json(directory / "run_status.json")
    manifest = read_json(directory / "artifact_manifest.json")
    require_equal(status["run_id"], run_id, f"{run_id} status ID")
    require_equal(status["status"], "complete", f"{run_id} status")
    require_equal(status["error"], None, f"{run_id} error")
    require_equal(manifest["run_id"], run_id, f"{run_id} manifest ID")
    require_equal(manifest["run_status"], "complete", f"{run_id} manifest status")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Empty manifest: {run_id}")
    observed: dict[str, str] = {}
    for record in files:
        relative = record["relative_to_run"]
        if relative in observed:
            raise ValueError(f"Duplicate manifest entry: {run_id}/{relative}")
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing/escaping manifest path: {run_id}/{relative}")
        require_equal(path.stat().st_size, record["size_bytes"], f"{run_id}/{relative} size")
        actual_hash = sha256_file(path)
        require_equal(actual_hash, record["sha256"], f"{run_id}/{relative} hash")
        observed[relative] = actual_hash
    for required in (
        "config.json", "metrics.json", "predictions.csv", "history.json",
        "checkpoint_metadata.json", "checkpoints/best_checkpoint.pt",
    ):
        if required not in observed:
            raise ValueError(f"Unmanifested required file: {run_id}/{required}")
    if "checkpoints/resume_checkpoint.pt" in observed or "traceback.txt" in observed:
        raise ValueError(f"Completed run contains resume/failure debris: {run_id}")
    return len(files), observed


def validate_new_predictions(frame: pd.DataFrame, lineage: pd.DataFrame, run_id: str) -> None:
    required = (
        "sample_id", "dataset_revision", "test_position", "source_row_index", "source",
        "strict_included", "strict_unique_included", "cleaned_text_sha256", "text",
        "true_id", "predicted_id",
    )
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f"{run_id} missing prediction columns: {missing}")
    expected_ids = [stable_winvoker_sample_id(value) for value in lineage["source_row_index"]]
    require_equal(frame["sample_id"].tolist(), expected_ids, f"{run_id} stable sample IDs")
    if not frame["sample_id"].is_unique:
        raise ValueError(f"Duplicate sample IDs: {run_id}")
    if not frame["dataset_revision"].eq(WINVOKER_DATASET_REVISION).all():
        raise ValueError(f"Dataset revision mismatch: {run_id}")
    for actual, expected in (
        ("test_position", "test_position"),
        ("source_row_index", "source_row_index"),
        ("source", "dataset"),
        ("cleaned_text_sha256", "cleaned_text_sha256"),
    ):
        if not np.array_equal(frame[actual].to_numpy(), lineage[expected].to_numpy()):
            raise ValueError(f"Ordered lineage mismatch: {run_id}/{actual}")
    for key in ("strict_included", "strict_unique_included"):
        actual = parse_boolean_series(frame[key], f"{run_id}.{key}").to_numpy()
        if not np.array_equal(actual, lineage[key].to_numpy()):
            raise ValueError(f"Strict-mask mismatch: {run_id}/{key}")


def validate_common_predictions(frame: pd.DataFrame, lineage: pd.DataFrame, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    require_equal(len(frame), len(lineage), f"{run_id} prediction row count")
    actual_hashes = frame["text"].astype(str).map(sha256_text).to_numpy()
    if not np.array_equal(actual_hashes, lineage["cleaned_text_sha256"].to_numpy()):
        raise ValueError(f"Canonical cleaned-text order mismatch: {run_id}")
    true = normalize_label_ids(frame["true_id"], f"{run_id}.true")
    pred = normalize_label_ids(frame["predicted_id"], f"{run_id}.predicted")
    if not np.array_equal(true, lineage["label_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"True-label order mismatch: {run_id}")
    for label_id, label_name in ID_TO_LABEL.items():
        if not frame.loc[frame["true_id"].eq(label_id), "true_label"].eq(label_name).all():
            raise ValueError(f"True-label names mismatch: {run_id}/{label_name}")
        if not frame.loc[frame["predicted_id"].eq(label_id), "predicted_label"].eq(label_name).all():
            raise ValueError(f"Predicted-label names mismatch: {run_id}/{label_name}")
    scores = frame[["logit_positive", "logit_negative", "logit_notr"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    if scores.shape != (len(lineage), 3) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid logit matrix: {run_id}")
    if not np.array_equal(np.argmax(scores, axis=1), pred):
        raise ValueError(f"Saved predictions disagree with argmax logits: {run_id}")
    return true, pred


def compare_metrics(
    true: np.ndarray, pred: np.ndarray, lineage: pd.DataFrame,
    stored: dict, run_id: str, *, historical: bool,
) -> list[dict]:
    strict = lineage["strict_included"].to_numpy(dtype=bool)
    result: list[dict] = []
    for subset, mask in (
        (SUBSETS[0], np.ones(len(lineage), dtype=bool)),
        (SUBSETS[1], strict),
    ):
        computed = classification_metrics(true[mask], pred[mask], subset_id=subset)
        if historical:
            reference = stored if subset == SUBSETS[0] else None
        else:
            reference = stored if subset == SUBSETS[0] else stored["views"][subset]
        if reference is not None:
            require_equal(computed["confusion_matrix"], reference["confusion_matrix"],
                          f"{run_id}/{subset} confusion")
            if "n" in reference:
                require_equal(computed["n"], reference["n"], f"{run_id}/{subset} support")
            for endpoint in ENDPOINTS:
                if not math.isclose(computed[endpoint], reference[endpoint], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{run_id}/{subset}/{endpoint} stored/recomputed mismatch")
        for endpoint in ENDPOINTS:
            result.append({
                "seed": int(run_id.split("seed")[1].split("_")[0]),
                "run_id": run_id,
                "provenance": "historical_conditional" if historical else "protocol_v2_migration",
                "subset_id": subset, "endpoint": endpoint, "value": computed[endpoint],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        parser.error(f"This frozen aggregation uses --run-id {RUN_ID}")
    preflight_four = read_json(PREFLIGHT_FOUR)
    preflight_seed42 = read_json(PREFLIGHT_SEED42)
    four_details = preflight_four["details"]
    seed42_details = preflight_seed42["details"]
    require_equal(preflight_four["checks_passed"], 22, "four-seed preflight passed checks")
    require_equal(preflight_four["checks_total"], 22, "four-seed preflight total checks")
    require_equal(preflight_seed42["checks_passed"], 23, "seed-42 extension preflight passed checks")
    require_equal(preflight_seed42["checks_total"], 23, "seed-42 extension preflight total checks")
    require_equal(sha256_file(LOCK), four_details["lock_config_sha256"], "original lock hash")
    require_equal(sha256_file(LOCK), seed42_details["lock_config_sha256"], "extended lock hash")
    require_equal(four_details["vocab_sha256"], seed42_details["vocab_sha256"], "shared vocabulary hash")
    require_equal(
        sha256_file(ROOT / "scripts/run_bilstm_protocol_v2.py"),
        seed42_details["runner_sha256"], "seed-42 runner hash",
    )
    audit = read_json(CHECKPOINT_AUDIT)
    require_equal(
        audit["input_hashes"]["checkpoint_sha256"],
        seed42_details["historical_checkpoint_sha256"], "historical checkpoint identity",
    )
    lineage = load_winvoker_test_lineage()
    output_dir = ROOT / "outputs/statistics" / args.run_id
    config = {
        "schema_version": "bilstm-five-seed-protocol-v2-v1.0",
        "run_id": args.run_id,
        "seeds": list(SEEDS),
        "original_four_seeds": list(MODERN_SEEDS),
        "historical_seed42_used_in_primary_cohort": False,
        "four_seed_preflight_results_sha256": sha256_file(PREFLIGHT_FOUR),
        "seed42_extension_preflight_results_sha256": sha256_file(PREFLIGHT_SEED42),
        "historical_audit_sha256": sha256_file(CHECKPOINT_AUDIT),
        "test_mask_sha256": sha256_file(ROOT / "outputs/audits/historic_split_lineage_v1/test_inclusion_mask.csv"),
    }
    with ExperimentRun(
        output_dir, run_id=args.run_id, run_kind="bilstm_five_seed_audit_aggregation",
        config=config, runtime_environment={"execution_kind": "cpu_only_postprocessing"},
    ) as run:
        inventory: list[dict] = []
        per_seed: list[dict] = []
        for seed in SEEDS:
            run_id = f"s2_neural_bilstm_winvoker_seed{seed}_protocol_v2"
            directory = ROOT / "outputs/experiments" / run_id
            count, hashes = verify_manifest(directory, run_id)
            manifest_sha256 = sha256_file(directory / "artifact_manifest.json")
            source_config = read_json(directory / "config.json")
            stored = read_json(directory / "metrics.json")
            history = read_json(directory / "history.json")
            metadata = read_json(directory / "checkpoint_metadata.json")
            expected_runner_sha256 = (
                seed42_details["runner_sha256"] if seed == 42 else four_details["runner_sha256"]
            )
            require_equal(source_config["training_seed"], seed, f"{run_id} seed")
            require_equal(source_config["resolved_device"], "cpu", f"{run_id} CPU device")
            require_equal(source_config["lock_config_sha256"], four_details["lock_config_sha256"], f"{run_id} lock")
            require_equal(source_config["source_hashes"]["runner"], expected_runner_sha256, f"{run_id} runner")
            require_equal(source_config["preprocessing"]["profile_id"], "winvoker_canonical_v1", f"{run_id} profile")
            require_equal(source_config["vocab_sha256"], four_details["vocab_sha256"], f"{run_id} vocabulary")
            require_equal(source_config["epochs_completed"], 3, f"{run_id} epochs")
            selected = history["val_f1"].index(max(history["val_f1"])) + 1
            require_equal(selected, source_config["selected_epoch_one_based"], f"{run_id} selected epoch")
            require_equal(selected - 1, metadata["selected_epoch_zero_based"], f"{run_id} checkpoint epoch")
            require_equal(metadata["sha256"], hashes["checkpoints/best_checkpoint.pt"], f"{run_id} checkpoint hash")
            validation_f1 = float(max(history["val_f1"]))
            require_equal(validation_f1, source_config["best_validation_weighted_f1"], f"{run_id} selected F1")
            frame = pd.read_csv(directory / "predictions.csv", keep_default_na=False)
            validate_new_predictions(frame, lineage, run_id)
            true, pred = validate_common_predictions(frame, lineage, run_id)
            per_seed.extend(compare_metrics(true, pred, lineage, stored, run_id, historical=False))
            inventory.append({
                "seed": seed, "run_id": run_id,
                "provenance": "protocol_v2_migration",
                "manifest_entries": count, "manifest_sha256": manifest_sha256,
                "prediction_rows": len(frame), "selected_epoch": selected,
                "validation_weighted_f1": validation_f1,
                "predictions_sha256": sha256_file(directory / "predictions.csv"),
            })
            print(f"Accepted {run_id}: {len(frame):,} ordered predictions")
        aggregates: list[dict] = []
        for cohort, seeds in (
            ("five_seed_protocol_v2", SEEDS),
            ("four_seed_original_cohort", MODERN_SEEDS),
        ):
            for subset in SUBSETS:
                for endpoint in ENDPOINTS:
                    values = [
                        row["value"] for row in per_seed
                        if row["seed"] in seeds and row["subset_id"] == subset
                        and row["endpoint"] == endpoint
                    ]
                    require_equal(len(values), len(seeds), f"{cohort}/{subset}/{endpoint} count")
                    aggregates.append({
                        "cohort": cohort, "subset_id": subset, "endpoint": endpoint,
                        "n_seeds": len(values), "mean": statistics.mean(values),
                        "sample_sd": statistics.stdev(values),
                        "min": min(values), "max": max(values),
                    })
        lines = [
            "# BiLSTM five-seed protocol-v2 audit and aggregate", "",
            "The primary cohort is five new protocol-v2 runs. The old seed-42 artifact is preserved separately as conditional historical continuity evidence. The seed-42 runner file hash differs from the four earlier migrations only because its CLI eligibility gate was extended; the training/evaluation function and numeric lock are unchanged. Do not describe the five source files as byte-identical.",
            "", "| Cohort | Test view | Endpoint | Mean | Sample SD |",
            "| :-- | :-- | :-- | --: | --: |",
        ]
        lines.extend(
            f"| {row['cohort']} | {row['subset_id']} | {row['endpoint']} | {row['mean']:.6f} | {row['sample_sd']:.6f} |"
            for row in aggregates
        )
        lines.extend(("", "The held-out test was not accessed by any training step in this aggregation; all metrics were recomputed from saved predictions. This command loads no model and performs no training.", ""))
        run.write_csv("input_inventory.csv", inventory)
        run.write_csv("per_seed_metrics.csv", per_seed)
        run.write_csv("aggregate_metrics.csv", aggregates)
        run.write_json("results.json", {
            "schema_version": config["schema_version"],
            "config": config,
            "input_inventory": inventory,
            "per_seed_metrics": per_seed,
            "aggregate_metrics": aggregates,
        })
        run.write_text("report.md", "\n".join(lines))
        run.write_text("notes.md", "# Notes\n\nCPU-only, no-training audit of five new protocol-v2 BiLSTM runs. Historical seed 42 is not part of the primary cohort.\n")
        run.declare_complete((
            "input_inventory.csv", "per_seed_metrics.csv", "aggregate_metrics.csv",
            "results.json", "report.md", "notes.md",
        ))


if __name__ == "__main__":
    main()
