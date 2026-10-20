"""Aggregate the five-seed weighted-versus-unweighted BERTurk ablation.

This is a CPU-only post-processing command. It validates all ten prediction
artifacts against the locked historical test lineage, recomputes complete and
strict metrics with the canonical metric implementation, and reports paired
training-seed uncertainty without treating test rows as independent runs.

The weighted experiments and seed-42 unweighted experiment are historical
artifacts whose resolved Hugging Face model revision was not recorded. The
four newer unweighted experiments are pinned protocol-v2 artifacts. The
result therefore retains an explicit conditional-provenance limitation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_constants import (
    BERTURK_MODEL_ID,
    BERTURK_MODEL_REVISION,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
    WINVOKER_SPLIT_SEED,
    WINVOKER_TEST_ROWS,
)
from src.protocol_metrics import classification_metrics
from src.protocol_predictions import (
    load_winvoker_test_lineage,
    normalize_label_ids,
    sha256_text,
)


SCHEMA_VERSION = "berturk-loss-ablation-v2.0"
SEEDS = (42, 52, 62, 72, 82)
VARIANTS = ("weighted", "unweighted")
SUBSETS = ("historic_test_complete", "historic_test_strict")
ENDPOINTS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "negative_precision",
    "negative_recall",
    "negative_f1",
)
EXPECTED_HYPERPARAMETERS = {
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "max_length": 128,
    "early_stopping_patience": 2,
}


def experiment_id(variant: str, seed: int) -> str:
    if variant == "weighted":
        return f"s1_berturk_berturk_winvoker_seed{seed}_weighted_valf1"
    if seed == 42:
        return "abl_berturk_winvoker_seed42_unweighted_valf1"
    return f"abl_berturk_winvoker_seed{seed}_unweighted_valf1_v2"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def config_value(config: dict[str, Any], legacy: str, protocol_v2: str) -> Any:
    if protocol_v2 in config:
        return config[protocol_v2]
    return config.get(legacy)


def scalar_close(actual: Any, expected: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_config(
    config: dict[str, Any],
    *,
    variant: str,
    seed: int,
    expected_id: str,
) -> list[str]:
    checks: list[str] = []
    recorded_id = config_value(config, "exp_id", "run_id")
    if recorded_id != expected_id:
        raise ValueError(f"Experiment ID mismatch for {expected_id}: {recorded_id!r}")
    checks.append("experiment_id")

    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Training seed mismatch for {expected_id}")
    if int(config.get("split_seed", -1)) != WINVOKER_SPLIT_SEED:
        raise ValueError(f"Split seed mismatch for {expected_id}")
    checks.extend(["training_seed", "split_seed"])

    model_id = config_value(config, "model_name", "model_id")
    dataset_id = config_value(config, "dataset", "dataset_id")
    if model_id != BERTURK_MODEL_ID:
        raise ValueError(f"Model ID mismatch for {expected_id}: {model_id!r}")
    if dataset_id != WINVOKER_DATASET_ID:
        raise ValueError(f"Dataset ID mismatch for {expected_id}: {dataset_id!r}")
    checks.extend(["model_id", "dataset_id"])

    recorded_loss = config.get("loss_mode")
    if recorded_loss != variant:
        documented_exception = variant == "weighted" and seed == 42 and recorded_loss is None
        if not documented_exception:
            raise ValueError(
                f"Loss mode mismatch for {expected_id}: expected {variant}, got {recorded_loss!r}"
            )
        checks.append("loss_mode_inferred_from_locked_seed42_experiment_id")
    else:
        checks.append("loss_mode")

    for key, expected in EXPECTED_HYPERPARAMETERS.items():
        recorded = config.get("epochs_maximum") if key == "epochs" else config.get(key)
        if recorded is None and key == "epochs":
            recorded = config.get("epochs")
        if recorded is None and key == "early_stopping_patience":
            recorded = config.get("early_stopping", {}).get("patience")
        if not scalar_close(recorded, expected):
            raise ValueError(
                f"Hyperparameter mismatch for {expected_id}: {key}={recorded!r}, expected {expected!r}"
            )
        checks.append(key)

    if config.get("schema_version") == "experiment-artifacts-v2.0":
        if config.get("model_revision") != BERTURK_MODEL_REVISION:
            raise ValueError(f"Pinned model revision mismatch for {expected_id}")
        if config.get("dataset_revision") != WINVOKER_DATASET_REVISION:
            raise ValueError(f"Pinned dataset revision mismatch for {expected_id}")
        if config.get("precision_mode") != "float32":
            raise ValueError(f"Precision-mode mismatch for {expected_id}")
        checks.extend(["model_revision", "dataset_revision", "precision_mode"])
    return checks


def verify_protocol_v2_manifest(experiment_dir: Path) -> tuple[int | None, str]:
    manifest_path = experiment_dir / "artifact_manifest.json"
    status_path = experiment_dir / "run_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        return None, "not_available_legacy_artifact"
    status = read_json(status_path)
    if status.get("status") != "complete":
        raise ValueError(f"Protocol-v2 experiment is not complete: {experiment_dir}")
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Malformed manifest: {manifest_path}")
    for item in files:
        relative = item.get("relative_to_run")
        if not relative:
            raise ValueError(f"Manifest entry lacks relative_to_run: {manifest_path}")
        path = experiment_dir / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Manifest size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Manifest SHA-256 mismatch: {path}")
    return len(files), "all_entries_match"


def load_aligned_predictions(
    prediction_path: Path,
    lineage: pd.DataFrame,
    *,
    expected_id: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(prediction_path, keep_default_na=False)
    required = {"text", "true_id", "predicted_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{expected_id} predictions lack columns: {sorted(missing)}")
    if len(frame) != WINVOKER_TEST_ROWS:
        raise ValueError(
            f"{expected_id} has {len(frame):,} predictions; expected {WINVOKER_TEST_ROWS:,}"
        )

    true_ids = normalize_label_ids(frame["true_id"], f"{expected_id}.true_id")
    predicted_ids = normalize_label_ids(frame["predicted_id"], f"{expected_id}.predicted_id")
    expected_true = lineage["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(true_ids, expected_true):
        position = int(np.flatnonzero(true_ids != expected_true)[0])
        raise ValueError(f"True-label lineage mismatch for {expected_id} at {position}")

    expected_hashes = lineage["cleaned_text_sha256"].astype(str).to_numpy()
    if "cleaned_text_sha256" in frame.columns:
        actual_hashes = frame["cleaned_text_sha256"].astype(str).to_numpy()
        hash_source = "recorded_cleaned_text_sha256"
    else:
        actual_hashes = frame["text"].astype(str).map(sha256_text).to_numpy()
        hash_source = "recomputed_sha256_from_legacy_text"
    if not np.array_equal(actual_hashes, expected_hashes):
        position = int(np.flatnonzero(actual_hashes != expected_hashes)[0])
        raise ValueError(f"Cleaned-text lineage mismatch for {expected_id} at {position}")

    if "test_position" in frame.columns:
        positions = pd.to_numeric(frame["test_position"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(positions, np.arange(WINVOKER_TEST_ROWS, dtype=np.int64)):
            raise ValueError(f"Noncanonical test ordering for {expected_id}")
    if "sample_id" in frame.columns and not frame["sample_id"].is_unique:
        raise ValueError(f"Duplicate stable sample IDs for {expected_id}")
    return true_ids, predicted_ids, hash_source


def endpoint_values(metrics: dict[str, Any]) -> dict[str, float]:
    negative = metrics["per_class"]["Negative"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "negative_precision": float(negative["precision"]),
        "negative_recall": float(negative["recall"]),
        "negative_f1": float(negative["f1"]),
    }


def verify_stored_metrics(
    stored: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    expected_id: str,
    subset_id: str,
) -> None:
    if subset_id == "historic_test_complete":
        reference = stored
    else:
        reference = stored.get("views", {}).get(subset_id)
        if reference is None:
            return
    for endpoint in ("accuracy", "macro_f1", "weighted_f1", "negative_f1"):
        if not math.isclose(
            float(reference[endpoint]),
            float(recomputed[endpoint]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Stored/recomputed {endpoint} mismatch for {expected_id}/{subset_id}"
            )


def sample_sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def exact_paired_seed_bootstrap(deltas: np.ndarray) -> dict[str, Any]:
    count = len(deltas)
    indices = np.asarray(list(itertools.product(range(count), repeat=count)), dtype=np.int64)
    means = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "method": "exact enumeration of paired-seed bootstrap resamples",
        "resampling_unit": "paired training seed",
        "enumerated_resamples": int(count**count),
        "confidence_level": 0.95,
        "interval": "percentile",
        "ci_low": float(low),
        "ci_high": float(high),
    }


def exact_sign_flip_test(deltas: np.ndarray) -> dict[str, Any]:
    count = len(deltas)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=count)))
    null_statistics = np.abs((signs * deltas).mean(axis=1))
    observed = abs(float(deltas.mean()))
    extreme = int(np.sum(null_statistics >= observed - 1e-15))
    assignments = int(2**count)
    return {
        "method": "exact two-sided paired-seed sign-flip permutation test",
        "statistic": "absolute mean paired delta",
        "enumerated_assignments": assignments,
        "extreme_assignments": extreme,
        "p_value": float(extreme / assignments),
        "grid_resolution": float(1.0 / assignments),
        "minimum_two_sided_p_value": float(2.0 / assignments),
    }


def render_report(
    aggregate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Five-Seed BERTurk Loss Ablation",
        "",
        "- Scope: exploratory paired weighted-versus-unweighted cross-entropy comparison.",
        "- Fixed training seeds: `42, 52, 62, 72, 82`; paired delta is `weighted - unweighted`.",
        "- Primary interpretation endpoint: Macro F1; key minority endpoint: Negative F1; continuity endpoint: Weighted F1.",
        "- Seed uncertainty: sample SD plus an exact 3,125-resample paired-seed percentile bootstrap interval.",
        "- Small-sample test: exact enumeration of all 32 paired sign assignments; the minimum attainable two-sided p-value is 0.0625.",
        "",
        "> Provenance limitation: all historical weighted runs and the historical unweighted seed-42 run lack a recorded resolved model revision. Unweighted seeds 52/62/72/82 are pinned to the declared protocol-v2 revision. Prediction lineage and the recorded training configuration are validated, but snapshot identity cannot be retroactively proven; conclusions are therefore conditional and exploratory.",
        "",
        "## Aggregate Metrics",
        "",
        "| Subset | Variant | Endpoint | Mean | Sample SD |",
        "| :-- | :-- | :-- | --: | --: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['subset_id']} | {row['variant']} | {row['endpoint']} | "
            f"{row['mean']:.6f} | {row['sample_sd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Seed Effects",
            "",
            "Negative deltas favor unweighted CE.",
            "",
            "| Subset | Endpoint | Mean Δ | Median Δ | Range | 95% paired-seed bootstrap CI | Exact sign-flip p | Weighted wins |",
            "| :-- | :-- | --: | --: | :-- | :-- | --: | --: |",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['subset_id']} | {row['endpoint']} | {row['delta_mean']:.6f} | "
            f"{row['delta_median']:.6f} | [{row['delta_min']:.6f}, {row['delta_max']:.6f}] | "
            f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}] | "
            f"{row['sign_flip_p_value']:.6f} | {row['weighted_wins']}/5 |"
        )
    legacy_count = sum(row["provenance_tier"] == "legacy_aligned_unknown_snapshot" for row in input_rows)
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            f"- `{legacy_count}` of `{len(input_rows)}` inputs are aligned historical artifacts with unknown resolved model snapshots.",
            "- The five seeds quantify training-run variability on one fixed benchmark; they are not independent datasets.",
            "- With five pairs, failure to reject cannot establish equivalence. Effect magnitude, direction consistency, interval width, and provenance must be interpreted together.",
            "- Complete-test results preserve historical continuity. Strict results exclude the prespecified exact-leakage/empty/conflicting-label rows.",
            "- This loss ablation is outside the protocol-v2 final confirmatory competitor family and must be labeled exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the five-seed BERTurk weighted/unweighted loss ablation"
    )
    parser.add_argument(
        "--run-id",
        default="berturk_loss_ablation_five_seed_v1",
        help="Unique output directory name under outputs/statistics",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = ROOT_DIR / "outputs" / "statistics" / args.run_id
    lineage = load_winvoker_test_lineage()
    strict_mask = lineage["strict_included"].to_numpy(dtype=bool)
    subset_masks = {
        "historic_test_complete": np.ones(len(lineage), dtype=bool),
        "historic_test_strict": strict_mask,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "analysis_scope": "exploratory_loss_ablation",
        "paired_delta_direction": "weighted_minus_unweighted",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "subsets": list(SUBSETS),
        "endpoints": list(ENDPOINTS),
        "dataset_id": WINVOKER_DATASET_ID,
        "dataset_revision": WINVOKER_DATASET_REVISION,
        "split_seed": WINVOKER_SPLIT_SEED,
        "lineage_path": str(
            (ROOT_DIR / "outputs/audits/historic_split_lineage_v1/test_inclusion_mask.csv")
            .relative_to(ROOT_DIR)
        ),
        "lineage_sha256": sha256_file(
            ROOT_DIR / "outputs/audits/historic_split_lineage_v1/test_inclusion_mask.csv"
        ),
        "provenance_limitation": (
            "Historical weighted runs and historical unweighted seed 42 do not record "
            "their resolved model revision; protocol-v2 unweighted seeds 52/62/72/82 "
            f"are pinned to {BERTURK_MODEL_REVISION}."
        ),
    }

    with ExperimentRun(
        output_dir,
        run_id=args.run_id,
        run_kind="berturk_loss_ablation_aggregation",
        config=config,
        runtime_environment={"execution_kind": "cpu_only_postprocessing"},
    ) as run:
        print("Loading and validating ten paired BERTurk prediction artifacts")
        input_rows: list[dict[str, Any]] = []
        per_seed_rows: list[dict[str, Any]] = []

        for variant in VARIANTS:
            for seed in SEEDS:
                exp_id = experiment_id(variant, seed)
                exp_dir = ROOT_DIR / "outputs" / "experiments" / exp_id
                config_path = exp_dir / "config.json"
                metrics_path = exp_dir / "metrics.json"
                prediction_path = exp_dir / "predictions.csv"
                for required_path in (config_path, metrics_path, prediction_path):
                    if not required_path.is_file():
                        raise FileNotFoundError(f"Missing required input: {required_path}")

                exp_config = read_json(config_path)
                checks = validate_config(
                    exp_config,
                    variant=variant,
                    seed=seed,
                    expected_id=exp_id,
                )
                manifest_entries, manifest_status = verify_protocol_v2_manifest(exp_dir)
                true_ids, predicted_ids, hash_source = load_aligned_predictions(
                    prediction_path,
                    lineage,
                    expected_id=exp_id,
                )
                stored_metrics = read_json(metrics_path)
                provenance_tier = (
                    "protocol_v2_pinned"
                    if exp_config.get("schema_version") == "experiment-artifacts-v2.0"
                    else "legacy_aligned_unknown_snapshot"
                )
                input_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "experiment_id": exp_id,
                        "provenance_tier": provenance_tier,
                        "model_revision": exp_config.get("model_revision"),
                        "config_path": config_path.relative_to(ROOT_DIR).as_posix(),
                        "config_sha256": sha256_file(config_path),
                        "predictions_path": prediction_path.relative_to(ROOT_DIR).as_posix(),
                        "predictions_sha256": sha256_file(prediction_path),
                        "prediction_rows": len(predicted_ids),
                        "lineage_hash_source": hash_source,
                        "manifest_entries_verified": manifest_entries,
                        "manifest_status": manifest_status,
                        "config_checks": "|".join(checks),
                    }
                )

                for subset_id in SUBSETS:
                    mask = subset_masks[subset_id]
                    computed = classification_metrics(
                        true_ids[mask],
                        predicted_ids[mask],
                        subset_id=subset_id,
                        interpretation="loss_ablation_aggregation",
                    )
                    verify_stored_metrics(
                        stored_metrics,
                        computed,
                        expected_id=exp_id,
                        subset_id=subset_id,
                    )
                    values = endpoint_values(computed)
                    per_seed_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "experiment_id": exp_id,
                            "subset_id": subset_id,
                            "n": int(computed["n"]),
                            **values,
                        }
                    )
                print(f"  accepted input: {exp_id}")

        per_seed = pd.DataFrame(per_seed_rows)
        aggregate_rows: list[dict[str, Any]] = []
        paired_delta_rows: list[dict[str, Any]] = []
        paired_summary_rows: list[dict[str, Any]] = []

        for subset_id in SUBSETS:
            for variant in VARIANTS:
                selected = per_seed[
                    (per_seed["subset_id"] == subset_id)
                    & (per_seed["variant"] == variant)
                ].sort_values("seed")
                if selected["seed"].tolist() != list(SEEDS):
                    raise ValueError(f"Incomplete seed set for {variant}/{subset_id}")
                for endpoint in ENDPOINTS:
                    values = selected[endpoint].to_numpy(dtype=np.float64)
                    aggregate_rows.append(
                        {
                            "subset_id": subset_id,
                            "variant": variant,
                            "endpoint": endpoint,
                            "n_seeds": len(values),
                            "mean": float(values.mean()),
                            "sample_sd": sample_sd(values),
                            "minimum": float(values.min()),
                            "maximum": float(values.max()),
                        }
                    )

            weighted = per_seed[
                (per_seed["subset_id"] == subset_id)
                & (per_seed["variant"] == "weighted")
            ].sort_values("seed")
            unweighted = per_seed[
                (per_seed["subset_id"] == subset_id)
                & (per_seed["variant"] == "unweighted")
            ].sort_values("seed")
            if weighted["seed"].tolist() != unweighted["seed"].tolist():
                raise ValueError(f"Paired seed mismatch for {subset_id}")

            for endpoint in ENDPOINTS:
                weighted_values = weighted[endpoint].to_numpy(dtype=np.float64)
                unweighted_values = unweighted[endpoint].to_numpy(dtype=np.float64)
                deltas = weighted_values - unweighted_values
                bootstrap = exact_paired_seed_bootstrap(deltas)
                sign_flip = exact_sign_flip_test(deltas)
                for position, seed in enumerate(SEEDS):
                    paired_delta_rows.append(
                        {
                            "subset_id": subset_id,
                            "seed": seed,
                            "endpoint": endpoint,
                            "weighted_value": float(weighted_values[position]),
                            "unweighted_value": float(unweighted_values[position]),
                            "delta_weighted_minus_unweighted": float(deltas[position]),
                        }
                    )
                paired_summary_rows.append(
                    {
                        "subset_id": subset_id,
                        "endpoint": endpoint,
                        "n_seed_pairs": len(deltas),
                        "delta_direction": "weighted_minus_unweighted",
                        "delta_mean": float(deltas.mean()),
                        "delta_sample_sd": sample_sd(deltas),
                        "delta_median": float(np.median(deltas)),
                        "delta_min": float(deltas.min()),
                        "delta_max": float(deltas.max()),
                        "weighted_wins": int(np.sum(deltas > 0.0)),
                        "ties": int(np.sum(deltas == 0.0)),
                        "unweighted_wins": int(np.sum(deltas < 0.0)),
                        "bootstrap_method": bootstrap["method"],
                        "bootstrap_resamples": bootstrap["enumerated_resamples"],
                        "bootstrap_ci_low": bootstrap["ci_low"],
                        "bootstrap_ci_high": bootstrap["ci_high"],
                        "sign_flip_method": sign_flip["method"],
                        "sign_flip_assignments": sign_flip["enumerated_assignments"],
                        "sign_flip_extreme_assignments": sign_flip["extreme_assignments"],
                        "sign_flip_p_value": sign_flip["p_value"],
                        "minimum_two_sided_p_value": sign_flip[
                            "minimum_two_sided_p_value"
                        ],
                    }
                )

        result = {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "inputs": input_rows,
            "per_seed_metrics": per_seed_rows,
            "aggregate_metrics": aggregate_rows,
            "paired_seed_deltas": paired_delta_rows,
            "paired_seed_summary": paired_summary_rows,
        }
        run.write_csv("input_inventory.csv", input_rows)
        run.write_csv("per_seed_metrics.csv", per_seed_rows)
        run.write_csv("aggregate_metrics.csv", aggregate_rows)
        run.write_csv("paired_seed_deltas.csv", paired_delta_rows)
        run.write_csv("paired_seed_summary.csv", paired_summary_rows)
        run.write_json("results.json", result)
        run.write_text(
            "report.md",
            render_report(aggregate_rows, paired_summary_rows, input_rows),
        )
        run.write_text(
            "notes.md",
            "# Notes\n\n"
            "CPU-only aggregation of the locked five paired training seeds. "
            "No model was loaded and no training or inference was performed.\n",
        )
        run.declare_complete(
            [
                "input_inventory.csv",
                "per_seed_metrics.csv",
                "aggregate_metrics.csv",
                "paired_seed_deltas.csv",
                "paired_seed_summary.csv",
                "results.json",
                "report.md",
                "notes.md",
            ]
        )

    print(f"Complete. Review: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
