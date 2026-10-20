"""Protocol-v2 paired statistical analysis for saved prediction artifacts.

The default suite is intentionally exploratory continuity evidence. It compares
the representative seed-42 weighted BERTurk run with the seed-42 Linear SVM
and BiLSTM runs on the strict historical test subset. It also reports the five
matched BERTurk/Linear-SVM seed deltas separately from example-level
uncertainty.

Example (run from the repository root inside the project container):

    python scripts/significance_tests.py

Custom exploratory comparisons can be supplied with ``--exp-a`` and
``--exp-b``. The script writes a self-contained, checksummed result directory
under ``outputs/statistics/`` by default.

The final confirmatory family is supplied through ``--comparison-config``.
That immutable JSON contract freezes the named comparisons, matched seed
series, endpoint family, subset, resampling settings, and random seed together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "outputs" / "experiments"
DEFAULT_LINEAGE_DIR = ROOT_DIR / "outputs" / "audits" / "historic_split_lineage_v1"
DEFAULT_METRICS = ["macro_f1", "negative_f1", "weighted_f1"]
ALLOWED_METRICS = ["accuracy", "weighted_f1", "macro_f1", "negative_f1"]
LABEL_IDS = (0, 1, 2)
ID_TO_LABEL = {0: "Positive", 1: "Negative", 2: "Notr"}
SCHEMA_VERSION = "significance-analysis-v2.1"
COMPARISON_CONFIG_SCHEMA_VERSION = "significance-comparison-config-v1"
DEFAULT_SEEDS = (42, 52, 62, 72, 82)


@dataclass(frozen=True)
class PredictionData:
    experiment_id: str
    path: Path
    file_sha256: str
    y_true: np.ndarray
    y_pred: np.ndarray
    cleaned_text_sha256: tuple[str, ...]


@dataclass(frozen=True)
class LineageData:
    path: Path
    file_sha256: str
    test_position: np.ndarray
    label_id: np.ndarray
    label: tuple[str, ...]
    source: tuple[str, ...]
    cleaned_text_sha256: tuple[str, ...]
    strict_included: np.ndarray
    strict_unique_included: np.ndarray


class Tee:
    """Write progress to both the terminal and the durable run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prediction_vector_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean {value!r} in {field} at CSV row {row_number}")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def environment_payload() -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "git_commit": git_commit(),
    }


def default_comparisons() -> list[tuple[str, str]]:
    return [
        (
            "s1_berturk_berturk_winvoker_seed42_weighted_valf1",
            "s1_tfidf_svm_winvoker_seed42_base",
        ),
        (
            "s1_berturk_berturk_winvoker_seed42_weighted_valf1",
            "s2_neural_bilstm_winvoker_seed42_base",
        ),
    ]


def default_seed_series() -> list[dict[str, Any]]:
    return [
        {
            "series_id": "weighted_berturk_vs_linear_svm",
            "description": "Matched declared seeds; BERTurk varies by training seed and Linear SVM is deterministic.",
            "pairs": [
                {
                    "seed": seed,
                    "exp_a": f"s1_berturk_berturk_winvoker_seed{seed}_weighted_valf1",
                    "exp_b": f"s1_tfidf_svm_winvoker_seed{seed}_base",
                }
                for seed in DEFAULT_SEEDS
            ],
        }
    ]


def load_comparison_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Comparison config must contain one JSON object.")
    if payload.get("schema_version") != COMPARISON_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported comparison-config schema_version: "
            f"{payload.get('schema_version')!r}"
        )

    required = {
        "analysis_id",
        "analysis_scope",
        "subset",
        "stratify",
        "metrics",
        "n_bootstrap",
        "n_randomizations",
        "randomization_batch_size",
        "confidence_level",
        "base_seed",
        "comparisons",
        "seed_series",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Comparison config is missing fields: {sorted(missing)}")
    if not isinstance(payload["analysis_id"], str) or not payload["analysis_id"]:
        raise ValueError("Comparison config requires a non-empty analysis_id.")
    if payload["analysis_scope"] not in {"confirmatory", "reviewer_requested_sensitivity"}:
        raise ValueError("Unsupported comparison-contract analysis scope.")
    if payload["subset"] not in {"strict", "full", "strict_unique"}:
        raise ValueError(f"Unsupported subset: {payload['subset']!r}")
    if payload["stratify"] not in {"source_label", "label", "none"}:
        raise ValueError(f"Unsupported stratification: {payload['stratify']!r}")
    metrics = payload["metrics"]
    if not isinstance(metrics, list) or not metrics or any(
        metric not in ALLOWED_METRICS for metric in metrics
    ):
        raise ValueError("Comparison config contains an empty or unsupported metric list.")
    if len(set(metrics)) != len(metrics):
        raise ValueError("Comparison-config metrics cannot contain duplicates.")
    if int(payload["n_bootstrap"]) < 100 or int(payload["n_randomizations"]) < 100:
        raise ValueError("Comparison config requires at least 100 bootstrap/randomization draws.")
    if int(payload["randomization_batch_size"]) < 1:
        raise ValueError("Comparison-config randomization_batch_size must be positive.")
    if not 0.0 < float(payload["confidence_level"]) < 1.0:
        raise ValueError("Comparison-config confidence_level must be between zero and one.")

    comparisons = payload["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("Comparison config must contain at least one named comparison.")
    comparison_ids: set[str] = set()
    comparison_pairs: set[tuple[str, str]] = set()
    for item in comparisons:
        if not isinstance(item, dict):
            raise ValueError("Each comparison must be a JSON object.")
        if not all(
            isinstance(item.get(field), str) and item[field]
            for field in ("comparison_id", "exp_a", "exp_b")
        ):
            raise ValueError(
                "Each comparison requires non-empty comparison_id, exp_a, and exp_b."
            )
        if item["comparison_id"] in comparison_ids:
            raise ValueError(f"Duplicate comparison_id: {item['comparison_id']}")
        pair = (item["exp_a"], item["exp_b"])
        if pair[0] == pair[1] or pair in comparison_pairs:
            raise ValueError(f"Invalid or duplicate comparison pair: {pair}")
        comparison_ids.add(item["comparison_id"])
        comparison_pairs.add(pair)

    series_specs = payload["seed_series"]
    if payload["analysis_scope"] == "reviewer_requested_sensitivity":
        if series_specs != []:
            raise ValueError("This fitted-model sensitivity must not include seed-series results.")
        if payload.get("anchor", {}).get("model") != "standard-CE BERTurk":
            raise ValueError("Sensitivity anchor must explicitly identify standard-CE BERTurk.")
        payload["comparison_config_path"] = str(path.relative_to(ROOT_DIR))
        payload["comparison_config_sha256"] = sha256_file(path)
        return payload
    if not isinstance(series_specs, list) or not series_specs:
        raise ValueError("Comparison config must contain matched seed-series specifications.")
    series_ids: set[str] = set()
    for series in series_specs:
        if not isinstance(series, dict):
            raise ValueError("Each seed-series specification must be a JSON object.")
        if not isinstance(series.get("series_id"), str) or not series["series_id"]:
            raise ValueError("Each seed series requires a non-empty series_id.")
        if series["series_id"] in series_ids:
            raise ValueError(f"Duplicate series_id: {series['series_id']}")
        if not isinstance(series.get("description"), str) or not series["description"]:
            raise ValueError(f"Seed series {series['series_id']} requires a description.")
        pairs = series.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"Seed series {series['series_id']} has no pairs.")
        seeds: set[int] = set()
        for pair in pairs:
            if not isinstance(pair, dict) or not all(
                field in pair for field in ("seed", "exp_a", "exp_b")
            ):
                raise ValueError(
                    f"Seed series {series['series_id']} contains an invalid pair."
                )
            seed = int(pair["seed"])
            if seed in seeds:
                raise ValueError(f"Duplicate seed {seed} in series {series['series_id']}.")
            if not isinstance(pair["exp_a"], str) or not isinstance(pair["exp_b"], str):
                raise ValueError(f"Non-string experiment ID in series {series['series_id']}.")
            if pair["exp_a"] == pair["exp_b"]:
                raise ValueError(f"Self-comparison in series {series['series_id']}.")
            seeds.add(seed)
        if seeds != set(DEFAULT_SEEDS):
            raise ValueError(
                f"Seed series {series['series_id']} must use exactly {list(DEFAULT_SEEDS)}; "
                f"found {sorted(seeds)}."
            )
        series_ids.add(series["series_id"])

    if series_ids != comparison_ids:
        raise ValueError(
            "Comparison and seed-series IDs must match exactly; "
            f"comparisons={sorted(comparison_ids)}, series={sorted(series_ids)}."
        )
    comparison_by_id = {item["comparison_id"]: item for item in comparisons}
    for series in series_specs:
        representative = next(pair for pair in series["pairs"] if int(pair["seed"]) == 42)
        comparison = comparison_by_id[series["series_id"]]
        if (representative["exp_a"], representative["exp_b"]) != (
            comparison["exp_a"],
            comparison["exp_b"],
        ):
            raise ValueError(
                f"Seed-42 pair does not match named comparison {series['series_id']}."
            )

    payload["comparison_config_path"] = str(path.relative_to(ROOT_DIR))
    payload["comparison_config_sha256"] = sha256_file(path)
    return payload


def sensitivity_input_gate(contract: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Freeze the existing evidence; never write into its directories."""
    reference = ROOT_DIR / "outputs/statistics/final_confirmatory_comparisons_v1"
    audit = ROOT_DIR / "outputs/audits/final_statistics_independent_audit_v1"
    preserved: dict[str, str] = {}
    for directory in (reference, audit):
        if json.loads((directory / "run_status.json").read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError(f"Preserved statistics are not complete: {directory}")
        manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = (directory / entry.get("relative_to_run", entry["path"])).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Preserved manifest path escapes its directory.")
            if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Preserved statistical artifact changed: {path}")
        for path in directory.rglob("*"):
            if path.is_file():
                preserved[str(path.relative_to(ROOT_DIR))] = sha256_file(path)
    original = json.loads((reference / "results.json").read_text(encoding="utf-8"))
    baseline = original["config"]
    original_contract = ROOT_DIR / "configs/statistics/final_confirmatory_comparisons_v1.json"
    if sha256_file(original_contract) != baseline["comparison_contract"]["comparison_config_sha256"]:
        raise ValueError("Original weighted comparison contract changed.")
    for key in ("subset", "stratify", "metrics", "n_bootstrap", "n_randomizations",
                "randomization_batch_size", "confidence_level", "base_seed"):
        if contract[key] != baseline[key]:
            raise ValueError(f"Sensitivity changed the accepted statistical method: {key}")
    anchor = "abl_berturk_winvoker_seed42_unweighted_valf1"
    expected_pairs = [(anchor, item["exp_b"]) for item in baseline["comparisons"]]
    if [(item["exp_a"], item["exp_b"]) for item in contract["comparisons"]] != expected_pairs:
        raise ValueError("Sensitivity must retain the exact five original competitor fits/order.")
    pins = {item["experiment_id"]: item["sha256"] for item in original["inputs"]}
    pins[anchor] = contract["historical_anchor_predictions_sha256"]
    for experiment_id in {value for pair in expected_pairs for value in pair}:
        path = EXPERIMENTS_DIR / experiment_id / "predictions.csv"
        digest = sha256_file(path)
        if digest != pins[experiment_id]:
            raise ValueError(f"Prediction identity changed: {experiment_id}")
        preserved[str(path.relative_to(ROOT_DIR))] = digest
    for relative in (baseline["lineage_mask"], contract["comparison_config_path"],
                     "configs/statistics/final_confirmatory_comparisons_v1.json",
                     "scripts/significance_tests.py"):
        preserved[relative] = sha256_file(ROOT_DIR / relative)
    if preserved[baseline["lineage_mask"]] != contract["lineage_mask_sha256"]:
        raise ValueError("Sensitivity mask differs from the frozen canonical mask.")
    if baseline["lineage_mask_sha256"] != contract["lineage_mask_sha256"]:
        raise ValueError("Sensitivity mask differs from the original statistical analysis.")
    write_json(output_dir / "input_inventory.json", {
        "sha256_by_path": preserved,
        "reference_analysis": str(reference.relative_to(ROOT_DIR)),
        "prediction_inputs": 6,
        "training": False, "inference": False,
    })
    return preserved


def broadcast_experiment_lists(exp_a: Sequence[str], exp_b: Sequence[str]) -> list[tuple[str, str]]:
    if len(exp_a) == len(exp_b):
        pairs = list(zip(exp_a, exp_b))
    elif len(exp_a) == 1:
        pairs = [(exp_a[0], item) for item in exp_b]
    elif len(exp_b) == 1:
        pairs = [(item, exp_b[0]) for item in exp_a]
    else:
        raise ValueError(
            "Experiment lists must have the same length, or one side must have length 1 for broadcasting."
        )

    if any(a == b for a, b in pairs):
        raise ValueError("A comparison cannot use the same experiment on both sides.")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate experiment pairs are not allowed.")
    return pairs


def load_lineage(lineage_dir: Path) -> LineageData:
    path = lineage_dir / "test_inclusion_mask.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical test mask: {path}")

    positions: list[int] = []
    label_ids: list[int] = []
    labels: list[str] = []
    sources: list[str] = []
    hashes: list[str] = []
    strict: list[bool] = []
    strict_unique: list[bool] = []
    required = {
        "test_position",
        "label",
        "label_id",
        "dataset",
        "strict_included",
        "strict_unique_included",
        "cleaned_text_sha256",
    }

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical test mask is missing columns: {sorted(missing)}")
        for csv_row, row in enumerate(reader, start=2):
            position = int(row["test_position"])
            expected_position = len(positions)
            if position != expected_position:
                raise ValueError(
                    f"Non-contiguous test_position at CSV row {csv_row}: "
                    f"expected {expected_position}, found {position}"
                )
            label_id = int(row["label_id"])
            label = row["label"]
            if label_id not in LABEL_IDS or label != ID_TO_LABEL[label_id]:
                raise ValueError(
                    f"Unexpected label mapping at test position {position}: {label!r}/{label_id}"
                )
            positions.append(position)
            label_ids.append(label_id)
            labels.append(label)
            sources.append(row["dataset"])
            hashes.append(row["cleaned_text_sha256"])
            strict.append(parse_bool(row["strict_included"], "strict_included", csv_row))
            strict_unique.append(
                parse_bool(row["strict_unique_included"], "strict_unique_included", csv_row)
            )

    if not positions:
        raise ValueError(f"Canonical test mask has no rows: {path}")

    return LineageData(
        path=path,
        file_sha256=sha256_file(path),
        test_position=np.asarray(positions, dtype=np.int64),
        label_id=np.asarray(label_ids, dtype=np.int64),
        label=tuple(labels),
        source=tuple(sources),
        cleaned_text_sha256=tuple(hashes),
        strict_included=np.asarray(strict, dtype=bool),
        strict_unique_included=np.asarray(strict_unique, dtype=bool),
    )


def load_predictions(experiment_id: str, lineage: LineageData) -> PredictionData:
    path = EXPERIMENTS_DIR / experiment_id / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    true_ids: list[int] = []
    predicted_ids: list[int] = []
    text_hashes: list[str] = []
    required = {"text", "true_id", "predicted_id"}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{experiment_id} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader):
            true_id = int(row["true_id"])
            predicted_id = int(row["predicted_id"])
            if true_id not in LABEL_IDS or predicted_id not in LABEL_IDS:
                raise ValueError(
                    f"{experiment_id} has an out-of-range label at prediction row {row_number}"
                )
            true_ids.append(true_id)
            predicted_ids.append(predicted_id)
            text_hashes.append(sha256_text(row["text"]))

    expected_rows = len(lineage.test_position)
    if len(true_ids) != expected_rows:
        raise ValueError(
            f"Prediction length mismatch: {experiment_id} has {len(true_ids)} rows; "
            f"canonical test mask has {expected_rows}"
        )

    true_array = np.asarray(true_ids, dtype=np.int64)
    label_mismatch = np.flatnonzero(true_array != lineage.label_id)
    if len(label_mismatch):
        position = int(label_mismatch[0])
        raise ValueError(
            f"True-label lineage mismatch for {experiment_id} at test position {position}"
        )

    for position, (actual_hash, expected_hash) in enumerate(
        zip(text_hashes, lineage.cleaned_text_sha256)
    ):
        if actual_hash != expected_hash:
            raise ValueError(
                f"Cleaned-text lineage mismatch for {experiment_id} at test position {position}"
            )

    return PredictionData(
        experiment_id=experiment_id,
        path=path,
        file_sha256=sha256_file(path),
        y_true=true_array,
        y_pred=np.asarray(predicted_ids, dtype=np.int64),
        cleaned_text_sha256=tuple(text_hashes),
    )


def subset_mask(lineage: LineageData, subset: str) -> np.ndarray:
    if subset == "full":
        return np.ones(len(lineage.test_position), dtype=bool)
    if subset == "strict":
        return lineage.strict_included.copy()
    if subset == "strict_unique":
        return lineage.strict_unique_included.copy()
    raise ValueError(f"Unsupported subset: {subset}")


def confusion_flat(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    encoded = y_true * len(LABEL_IDS) + y_pred
    return np.bincount(encoded, minlength=len(LABEL_IDS) ** 2).astype(np.int64)


def metric_matrix_from_flat(flat_confusions: np.ndarray, metric: str) -> np.ndarray:
    flat = np.asarray(flat_confusions, dtype=np.float64)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    matrices = flat.reshape(-1, len(LABEL_IDS), len(LABEL_IDS))
    total = matrices.sum(axis=(1, 2))
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    support = matrices.sum(axis=2)
    predicted = matrices.sum(axis=1)
    denominator = support + predicted
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator != 0,
    )

    if metric == "accuracy":
        return np.divide(
            true_positive.sum(axis=1),
            total,
            out=np.zeros_like(total, dtype=np.float64),
            where=total != 0,
        )
    if metric == "weighted_f1":
        return np.divide(
            (per_class_f1 * support).sum(axis=1),
            total,
            out=np.zeros_like(total, dtype=np.float64),
            where=total != 0,
        )
    if metric == "macro_f1":
        return per_class_f1.mean(axis=1)
    if metric == "negative_f1":
        return per_class_f1[:, 1]
    raise ValueError(f"Unsupported metric: {metric}")


def metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(metric_matrix_from_flat(confusion_flat(y_true, y_pred), metric)[0])


def metric_values(flat_confusion: np.ndarray, metrics: Sequence[str]) -> dict[str, float]:
    return {
        metric: float(metric_matrix_from_flat(flat_confusion, metric)[0])
        for metric in metrics
    }


def stable_seed(base_seed: int, exp_a: str, exp_b: str, purpose: str) -> int:
    material = f"{base_seed}|{exp_a}|{exp_b}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def build_group_design(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    cleaned_hashes: Sequence[str],
    sources: Sequence[str],
    stratify: str,
) -> dict[str, Any]:
    group_lookup: dict[str, int] = {}
    group_hashes: list[str] = []
    row_group = np.empty(len(y_true), dtype=np.int64)
    group_sources: list[set[str]] = []
    group_labels: list[set[int]] = []

    for row_index, (text_hash, source, label_id) in enumerate(
        zip(cleaned_hashes, sources, y_true)
    ):
        group_index = group_lookup.get(text_hash)
        if group_index is None:
            group_index = len(group_hashes)
            group_lookup[text_hash] = group_index
            group_hashes.append(text_hash)
            group_sources.append(set())
            group_labels.append(set())
        row_group[row_index] = group_index
        group_sources[group_index].add(str(source))
        group_labels[group_index].add(int(label_id))

    n_groups = len(group_hashes)
    cells_a = y_true * len(LABEL_IDS) + pred_a
    cells_b = y_true * len(LABEL_IDS) + pred_b
    group_cm_a = np.bincount(
        row_group * (len(LABEL_IDS) ** 2) + cells_a,
        minlength=n_groups * (len(LABEL_IDS) ** 2),
    ).reshape(n_groups, -1)
    group_cm_b = np.bincount(
        row_group * (len(LABEL_IDS) ** 2) + cells_b,
        minlength=n_groups * (len(LABEL_IDS) ** 2),
    ).reshape(n_groups, -1)
    group_sizes = np.bincount(row_group, minlength=n_groups)

    strata_lists: dict[str, list[int]] = {}
    heterogeneous_source_groups = 0
    heterogeneous_label_groups = 0
    for group_index, (source_set, label_set) in enumerate(zip(group_sources, group_labels)):
        if len(source_set) == 1:
            source_value = next(iter(source_set))
        else:
            source_value = "<mixed-source>"
            heterogeneous_source_groups += 1
        if len(label_set) == 1:
            label_value = ID_TO_LABEL[next(iter(label_set))]
        else:
            label_value = "<mixed-label>"
            heterogeneous_label_groups += 1

        if stratify == "source_label":
            stratum = f"{source_value}|{label_value}"
        elif stratify == "label":
            stratum = label_value
        elif stratify == "none":
            stratum = "all"
        else:
            raise ValueError(f"Unsupported stratification: {stratify}")
        strata_lists.setdefault(stratum, []).append(group_index)

    strata = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in sorted(strata_lists.items())
    }
    rows_in_non_singleton_groups = int(group_sizes[group_sizes > 1].sum())
    diagnostics = {
        "n_rows": int(len(y_true)),
        "n_cleaned_text_groups": int(n_groups),
        "n_non_singleton_groups": int(np.sum(group_sizes > 1)),
        "rows_in_non_singleton_groups": rows_in_non_singleton_groups,
        "largest_group_rows": int(group_sizes.max()),
        "heterogeneous_source_groups": int(heterogeneous_source_groups),
        "heterogeneous_label_groups": int(heterogeneous_label_groups),
        "stratification": stratify,
        "n_strata": len(strata),
        "strata": [
            {
                "stratum": name,
                "groups": int(len(indices)),
                "rows": int(group_sizes[indices].sum()),
            }
            for name, indices in strata.items()
        ],
    }
    return {
        "group_cm_a": group_cm_a.astype(np.int64, copy=False),
        "group_cm_b": group_cm_b.astype(np.int64, copy=False),
        "group_sizes": group_sizes.astype(np.int64, copy=False),
        "strata": strata,
        "diagnostics": diagnostics,
    }


def paired_cluster_bootstrap(
    design: dict[str, Any],
    metrics: Sequence[str],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_cm_a = design["group_cm_a"]
    group_cm_b = design["group_cm_b"]
    strata = design["strata"]
    observed_a = metric_values(group_cm_a.sum(axis=0), metrics)
    observed_b = metric_values(group_cm_b.sum(axis=0), metrics)
    deltas = np.empty((n_bootstrap, len(metrics)), dtype=np.float64)
    sampled_row_counts = np.empty(n_bootstrap, dtype=np.int64)
    progress_interval = max(1, n_bootstrap // 10)

    for iteration in range(n_bootstrap):
        sampled_a = np.zeros(len(LABEL_IDS) ** 2, dtype=np.int64)
        sampled_b = np.zeros(len(LABEL_IDS) ** 2, dtype=np.int64)
        for group_indices in strata.values():
            if len(group_indices) == 1:
                selected = group_indices
            else:
                positions = rng.integers(0, len(group_indices), size=len(group_indices))
                selected = group_indices[positions]
            sampled_a += group_cm_a[selected].sum(axis=0)
            sampled_b += group_cm_b[selected].sum(axis=0)
        sampled_row_counts[iteration] = int(sampled_a.sum())
        for metric_index, metric in enumerate(metrics):
            deltas[iteration, metric_index] = float(
                metric_matrix_from_flat(sampled_a, metric)[0]
                - metric_matrix_from_flat(sampled_b, metric)[0]
            )
        if (iteration + 1) % progress_interval == 0 or iteration + 1 == n_bootstrap:
            print(f"  cluster bootstrap: {iteration + 1:,}/{n_bootstrap:,}")

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    endpoint_results = {}
    for metric_index, metric in enumerate(metrics):
        ci_low, ci_high = np.percentile(
            deltas[:, metric_index], [lower_percentile, upper_percentile]
        )
        endpoint_results[metric] = {
            "metric": metric,
            "exp_a_value": observed_a[metric],
            "exp_b_value": observed_b[metric],
            "delta_a_minus_b": observed_a[metric] - observed_b[metric],
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    stratification = design["diagnostics"]["stratification"]
    if stratification == "source_label":
        method = "source/label-stratified cleaned-text cluster bootstrap"
    elif stratification == "label":
        method = "label-stratified cleaned-text cluster bootstrap"
    else:
        method = "unstratified cleaned-text cluster bootstrap"
    return {
        "method": method,
        "interval": "percentile",
        "confidence_level": confidence_level,
        "n_resamples": n_bootstrap,
        "seed": seed,
        "sampled_row_count_mean": float(sampled_row_counts.mean()),
        "sampled_row_count_min": int(sampled_row_counts.min()),
        "sampled_row_count_max": int(sampled_row_counts.max()),
        "endpoints": endpoint_results,
    }


def paired_group_randomization(
    design: dict[str, Any],
    metrics: Sequence[str],
    n_randomizations: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_cm_a = design["group_cm_a"].astype(np.float64, copy=False)
    group_cm_b = design["group_cm_b"].astype(np.float64, copy=False)
    base_a = group_cm_a.sum(axis=0)
    base_b = group_cm_b.sum(axis=0)
    swap_change = group_cm_b - group_cm_a
    observed_delta = {
        metric: float(
            metric_matrix_from_flat(base_a, metric)[0]
            - metric_matrix_from_flat(base_b, metric)[0]
        )
        for metric in metrics
    }
    extreme_counts = {metric: 0 for metric in metrics}
    tolerance = 1e-15

    completed = 0
    progress_interval = max(1, n_randomizations // 10)
    next_progress = progress_interval
    while completed < n_randomizations:
        current_batch = min(batch_size, n_randomizations - completed)
        flips = rng.integers(
            0,
            2,
            size=(current_batch, len(group_cm_a)),
            dtype=np.int8,
        ).astype(np.float64)
        changes = flips @ swap_change
        randomized_a = base_a[np.newaxis, :] + changes
        randomized_b = base_b[np.newaxis, :] - changes
        for metric in metrics:
            randomized_delta = metric_matrix_from_flat(
                randomized_a, metric
            ) - metric_matrix_from_flat(randomized_b, metric)
            extreme_counts[metric] += int(
                np.sum(
                    np.abs(randomized_delta)
                    >= abs(observed_delta[metric]) - tolerance
                )
            )
        completed += current_batch
        if completed >= next_progress or completed == n_randomizations:
            print(f"  paired randomization: {completed:,}/{n_randomizations:,}")
            while next_progress <= completed:
                next_progress += progress_interval

    resolution = 1.0 / (n_randomizations + 1)
    endpoints = {}
    for metric in metrics:
        corrected_p = (extreme_counts[metric] + 1.0) / (n_randomizations + 1.0)
        endpoints[metric] = {
            "metric": metric,
            "observed_delta_a_minus_b": observed_delta[metric],
            "extreme_randomizations": int(extreme_counts[metric]),
            "p_value_raw": float(corrected_p),
            "p_value_resolution": float(resolution),
        }

    return {
        "method": "two-sided paired cleaned-text-group prediction-swap randomization",
        "finite_sample_correction": "plus-one: (extreme + 1) / (randomizations + 1)",
        "n_randomizations": n_randomizations,
        "seed": seed,
        "p_value_resolution": resolution,
        "endpoints": endpoints,
    }


def exact_two_sided_binomial(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {
            "p_value_exact": 1.0,
            "p_value_exact_scientific": "1.000000e+0",
            "log10_p_value_exact": 0.0,
        }

    tail_k = min(k, n - k)
    log_half = math.log(0.5)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(x + 1)
        - math.lgamma(n - x + 1)
        + n * log_half
        for x in range(tail_k + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(sum(math.exp(term - maximum) for term in log_terms))
    log_p = min(0.0, math.log(2.0) + log_tail)
    log10_p = log_p / math.log(10.0)
    exponent = math.floor(log10_p)
    mantissa = math.exp(log_p - exponent * math.log(10.0))
    p_float = math.exp(log_p) if log_p >= math.log(sys.float_info.min) else None
    return {
        "p_value_exact": float(p_float) if p_float is not None else None,
        "p_value_exact_scientific": f"{mantissa:.6f}e{exponent:+d}",
        "log10_p_value_exact": float(log10_p),
    }


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_only_correct = int(np.sum(correct_a & ~correct_b))
    b_only_correct = int(np.sum(~correct_a & correct_b))
    discordant_total = a_only_correct + b_only_correct
    difference = abs(a_only_correct - b_only_correct)
    chi2_cc = (
        max(difference - 1, 0) ** 2 / discordant_total
        if discordant_total
        else 0.0
    )
    exact = exact_two_sided_binomial(
        min(a_only_correct, b_only_correct), discordant_total
    )
    return {
        "unit": "test row",
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "discordant_total": discordant_total,
        "continuity_corrected_chi_square_diagnostic": float(chi2_cc),
        **exact,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * p_values[original_index]
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)
    return adjusted


def endpoint_role(metric: str) -> str:
    if metric == "macro_f1":
        return "primary"
    if metric in {"negative_f1", "weighted_f1"}:
        return "key_secondary"
    return "exploratory"


def analyze_seed_series(
    series_specs: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    mask: np.ndarray,
    get_prediction,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for series in series_specs:
        b_vector_hashes: set[str] = set()
        for pair in series["pairs"]:
            pred_a = get_prediction(pair["exp_a"])
            pred_b = get_prediction(pair["exp_b"])
            y_true = pred_a.y_true[mask]
            y_pred_a = pred_a.y_pred[mask]
            y_pred_b = pred_b.y_pred[mask]
            if not np.array_equal(y_true, pred_b.y_true[mask]):
                raise ValueError(f"Seed-series label mismatch: {pair['exp_a']} vs {pair['exp_b']}")
            b_vector_hashes.add(prediction_vector_sha256(y_pred_b))
            for metric in metrics:
                value_a = metric_value(metric, y_true, y_pred_a)
                value_b = metric_value(metric, y_true, y_pred_b)
                per_seed_rows.append(
                    {
                        "series_id": series["series_id"],
                        "seed": int(pair["seed"]),
                        "exp_a": pair["exp_a"],
                        "exp_b": pair["exp_b"],
                        "metric": metric,
                        "exp_a_value": value_a,
                        "exp_b_value": value_b,
                        "delta_a_minus_b": value_a - value_b,
                    }
                )

        for metric in metrics:
            metric_rows = [
                row
                for row in per_seed_rows
                if row["series_id"] == series["series_id"] and row["metric"] == metric
            ]
            deltas = np.asarray([row["delta_a_minus_b"] for row in metric_rows])
            values_a = np.asarray([row["exp_a_value"] for row in metric_rows])
            values_b = np.asarray([row["exp_b_value"] for row in metric_rows])
            summary_rows.append(
                {
                    "series_id": series["series_id"],
                    "description": series["description"],
                    "metric": metric,
                    "n_seed_pairs": int(len(deltas)),
                    "exp_a_mean": float(values_a.mean()),
                    "exp_a_sample_sd": float(values_a.std(ddof=1)) if len(values_a) > 1 else None,
                    "exp_b_mean": float(values_b.mean()),
                    "exp_b_sample_sd": float(values_b.std(ddof=1)) if len(values_b) > 1 else None,
                    "delta_mean": float(deltas.mean()),
                    "delta_median": float(np.median(deltas)),
                    "delta_min": float(deltas.min()),
                    "delta_max": float(deltas.max()),
                    "delta_range": float(deltas.max() - deltas.min()),
                    "exp_b_unique_prediction_vectors": int(len(b_vector_hashes)),
                    "interpretation": (
                        "descriptive paired training-seed variability; not an example-level confidence interval"
                    ),
                }
            )

    return per_seed_rows, summary_rows


def p_display(p_value: float, extremes: int, resolution: float) -> str:
    if extremes == 0:
        return f"{p_value:.6g} (0 extremes; +1 estimate; resolution {resolution:.6g})"
    return f"{p_value:.6g}"


def render_report(
    config: dict[str, Any],
    lineage_summary: dict[str, Any],
    summary_rows: Sequence[dict[str, Any]],
    comparison_results: Sequence[dict[str, Any]],
    seed_summary_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Protocol-v2 Statistical Analysis",
        "",
        f"- Analysis scope: `{config['analysis_scope']}`",
        f"- Confirmatory: `{str(config['confirmatory']).lower()}`",
        f"- Test subset: `{config['subset']}` (`{lineage_summary['included_rows']:,}` / `{lineage_summary['full_rows']:,}` rows)",
        f"- Resampling unit: cleaned-text group; stratification: `{config['stratify']}`",
        f"- Bootstrap: `{config['n_bootstrap']:,}` paired cluster resamples, percentile `{config['confidence_level']:.1%}` intervals",
        f"- Metric p-values: `{config['n_randomizations']:,}` two-sided paired group swaps with plus-one correction",
        "- Multiplicity: Holm adjustment separately within each metric endpoint family",
        "- McNemar: exact two-sided row-level test, reported once per named model pair",
        "",
    ]
    if config["analysis_scope"] == "reviewer_requested_sensitivity":
        lines.extend([
            "> Reviewer-requested, post-test sensitivity using historical standard-CE BERTurk seed 42 and the five original seed-42 competitors. This is not preregistered or confirmatory; the original weighted-CE confirmatory analysis remains separate and unchanged.",
            "",
            "Historical BERTurk's resolved model snapshot was not recorded. These results condition on the named fits and fixed corpus, not architecture alone or training-seed/split variability. No seed-series analysis is generated. Nonsignificance does not establish equivalence. A zero-exceedance plus-one Monte Carlo estimate of 1/5001 is the minimum resolution, not an upper bound on the exact probability.",
            "",
        ])
    elif not config["confirmatory"]:
        lines.extend(
            [
                "> These results are exploratory continuity evidence. The protocol-v2 confirmatory competitor family is incomplete until the validation-selected multilingual and eligible Turkish-specific Transformer finalists exist.",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Effects and Conditional Test-Sample Uncertainty",
            "",
            f"| Model A | Model B | Endpoint | A | B | Δ A−B | {config['confidence_level']:.1%} cluster-bootstrap CI | randomization p | Holm p |",
            "| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | --: |",
        ]
    )
    for row in summary_rows:
        raw_display = p_display(
            row["p_value_raw"],
            row["extreme_randomizations"],
            row["p_value_resolution"],
        )
        lines.append(
            f"| `{row['exp_a']}` | `{row['exp_b']}` | {row['metric']} ({row['endpoint_role']}) "
            f"| {row['exp_a_value']:.6f} | {row['exp_b_value']:.6f} | {row['delta_a_minus_b']:+.6f} "
            f"| [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {raw_display} | {row['p_value_holm']:.6g} |"
        )

    lines.extend(
        [
            "",
            "The p-values in this table come from the paired group-randomization test; the bootstrap supplies effect intervals. Both condition on the named fitted models and this fixed test corpus.",
            "",
            "## Exact McNemar Results",
            "",
            "| Model A | Model B | A only correct | B only correct | Discordant | Exact two-sided p |",
            "| :-- | :-- | --: | --: | --: | :-- |",
        ]
    )
    for comparison in comparison_results:
        mc = comparison["mcnemar"]
        lines.append(
            f"| `{comparison['exp_a']}` | `{comparison['exp_b']}` | {mc['a_only_correct']:,} "
            f"| {mc['b_only_correct']:,} | {mc['discordant_total']:,} | {mc['p_value_exact_scientific']} |"
        )

    lines.extend(["", "## Resampling Diagnostics", ""])
    for comparison in comparison_results:
        diagnostics = comparison["group_design"]
        lines.extend(
            [
                f"### `{comparison['exp_a']}` vs `{comparison['exp_b']}`",
                "",
                f"- Rows: `{diagnostics['n_rows']:,}`; cleaned-text groups: `{diagnostics['n_cleaned_text_groups']:,}`.",
                f"- Non-singleton groups: `{diagnostics['n_non_singleton_groups']:,}` containing `{diagnostics['rows_in_non_singleton_groups']:,}` rows; largest group: `{diagnostics['largest_group_rows']}` rows.",
                f"- Heterogeneous-source groups: `{diagnostics['heterogeneous_source_groups']}`; heterogeneous-label groups: `{diagnostics['heterogeneous_label_groups']}`.",
                f"- Strata: `{diagnostics['n_strata']}`. Mixed-source groups remain intact and enter a `<mixed-source>` label stratum.",
                "",
            ]
        )

    if seed_summary_rows:
        lines.extend(
            [
                "## Separate Five-Seed Variability Summary",
                "",
                "| Series | Endpoint | Seed pairs | A mean ± SD | B mean ± SD | Δ mean | Δ median | Δ range |",
                "| :-- | :-- | --: | :-- | :-- | --: | --: | :-- |",
            ]
        )
        for row in seed_summary_rows:
            lines.append(
                f"| `{row['series_id']}` | {row['metric']} | {row['n_seed_pairs']} "
                f"| {row['exp_a_mean']:.6f} ± {row['exp_a_sample_sd']:.6f} "
                f"| {row['exp_b_mean']:.6f} ± {row['exp_b_sample_sd']:.6f} "
                f"| {row['delta_mean']:+.6f} | {row['delta_median']:+.6f} "
                f"| [{row['delta_min']:+.6f}, {row['delta_max']:+.6f}] |"
            )
        lines.extend(
            [
                "",
                "This section describes paired training-seed deltas on a fixed split. It is not interchangeable with the cluster-bootstrap intervals above and does not establish split-to-split stability.",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- The strict subset removes known exact cleaned-text train/validation overlap, empty cleaned texts, and globally conflicting cleaned-text labels; it does not establish freedom from semantic near-duplicates.",
            "- Cleaned-text groups are kept intact in bootstrap resampling and paired randomization. Source/label strata are used for the bootstrap because source and label are strongly associated in this corpus.",
            "- Exact McNemar is a paired row-level correctness analysis. It is reported separately and once per model pair; it does not replace the grouped effect intervals.",
            "- Statistical evidence does not by itself establish practical importance, generalization to another dataset, or the cause of an observed model difference.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(output_dir: Path, status: str) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_status": status,
        "manifest_excludes": ["artifact_manifest.json"],
        "files": files,
    }


def run_analysis(args, output_dir: Path) -> dict[str, Any]:
    comparison_contract = None
    preserved_inputs = None
    comparison_ids: list[str] | None = None
    if args.comparison_config is not None:
        comparison_contract = load_comparison_config(args.comparison_config)
        args.analysis_scope = comparison_contract["analysis_scope"]
        args.subset = comparison_contract["subset"]
        args.stratify = comparison_contract["stratify"]
        args.metrics = comparison_contract["metrics"]
        args.n_bootstrap = int(comparison_contract["n_bootstrap"])
        args.n_randomizations = int(comparison_contract["n_randomizations"])
        args.randomization_batch_size = int(
            comparison_contract["randomization_batch_size"]
        )
        args.confidence_level = float(comparison_contract["confidence_level"])
        args.seed = int(comparison_contract["base_seed"])

    if args.analysis_scope == "reviewer_requested_sensitivity":
        if comparison_contract is None:
            raise ValueError("Reviewer sensitivity requires an immutable comparison contract.")
        expected_output = ROOT_DIR / "outputs/statistics" / comparison_contract["analysis_id"]
        if output_dir.resolve() != expected_output.resolve():
            raise ValueError("Sensitivity output must match the separate contract analysis ID.")
        preserved_inputs = sensitivity_input_gate(comparison_contract, output_dir)

    lineage = load_lineage(args.lineage_dir)
    mask = subset_mask(lineage, args.subset)
    included_positions = np.flatnonzero(mask)
    if preserved_inputs is not None:
        if lineage.file_sha256 != comparison_contract["lineage_mask_sha256"]:
            raise ValueError("Loaded sensitivity lineage does not match frozen mask.")
        if len(mask) != 88136 or int(mask.sum()) != 87632:
            raise ValueError("Sensitivity requires 88,136 full / 87,632 strict rows.")
    included_hashes = tuple(lineage.cleaned_text_sha256[index] for index in included_positions)
    included_sources = tuple(lineage.source[index] for index in included_positions)

    if args.comparison_config is not None:
        comparisons = [
            (item["exp_a"], item["exp_b"])
            for item in comparison_contract["comparisons"]
        ]
        comparison_ids = [
            item["comparison_id"] for item in comparison_contract["comparisons"]
        ]
        seed_series_specs = comparison_contract["seed_series"]
        using_defaults = False
    elif args.exp_a is None and args.exp_b is None:
        comparisons = default_comparisons()
        seed_series_specs = [] if args.skip_seed_summary else default_seed_series()
        using_defaults = True
    elif args.exp_a is None or args.exp_b is None:
        raise ValueError(
            "Provide both --exp-a and --exp-b, or provide neither to use the default suite."
        )
    else:
        comparisons = broadcast_experiment_lists(args.exp_a, args.exp_b)
        seed_series_specs = []
        using_defaults = False

    if args.analysis_scope == "confirmatory" and using_defaults:
        raise ValueError(
            "The default suite is not the complete protocol-v2 confirmatory family. "
            "Supply the validation-selected finalists explicitly with --exp-a/--exp-b."
        )

    prediction_cache: dict[str, PredictionData] = {}

    def get_prediction(experiment_id: str) -> PredictionData:
        if experiment_id not in prediction_cache:
            print(f"Validating prediction lineage: {experiment_id}")
            prediction_cache[experiment_id] = load_predictions(experiment_id, lineage)
        return prediction_cache[experiment_id]

    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "analysis_scope": args.analysis_scope,
        "confirmatory": args.analysis_scope == "confirmatory",
        "subset": args.subset,
        "stratify": args.stratify,
        "metrics": list(args.metrics),
        "endpoint_roles": {metric: endpoint_role(metric) for metric in args.metrics},
        "n_bootstrap": args.n_bootstrap,
        "n_randomizations": args.n_randomizations,
        "randomization_batch_size": args.randomization_batch_size,
        "confidence_level": args.confidence_level,
        "base_seed": args.seed,
        "multiplicity": "Holm within each metric endpoint family across named comparisons",
        "comparisons": [{"exp_a": a, "exp_b": b} for a, b in comparisons],
        "seed_series": seed_series_specs,
        "lineage_mask": str(lineage.path.relative_to(ROOT_DIR)),
        "lineage_mask_sha256": lineage.file_sha256,
    }
    if comparison_contract is not None:
        config["comparison_contract"] = comparison_contract
    write_json(output_dir / "config.json", config)

    print(f"Analysis scope: {args.analysis_scope}")
    print(f"Subset: {args.subset} ({int(mask.sum()):,}/{len(mask):,} rows)")
    print(f"Comparisons: {len(comparisons)}")
    print(
        f"Resampling: {args.n_bootstrap:,} cluster bootstraps and "
        f"{args.n_randomizations:,} group randomizations"
    )

    comparison_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_diagnostic_rows: list[dict[str, Any]] = []

    for pair_index, (exp_a, exp_b) in enumerate(comparisons, start=1):
        print(f"[{pair_index}/{len(comparisons)}] {exp_a} vs {exp_b}")
        data_a = get_prediction(exp_a)
        data_b = get_prediction(exp_b)
        y_true = data_a.y_true[mask]
        if not np.array_equal(y_true, data_b.y_true[mask]):
            raise ValueError(f"Aligned true labels differ: {exp_a} vs {exp_b}")
        pred_a = data_a.y_pred[mask]
        pred_b = data_b.y_pred[mask]
        design = build_group_design(
            y_true=y_true,
            pred_a=pred_a,
            pred_b=pred_b,
            cleaned_hashes=included_hashes,
            sources=included_sources,
            stratify=args.stratify,
        )
        bootstrap_seed = stable_seed(args.seed, exp_a, exp_b, "bootstrap")
        randomization_seed = stable_seed(args.seed, exp_a, exp_b, "randomization")
        bootstrap = paired_cluster_bootstrap(
            design=design,
            metrics=args.metrics,
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            seed=bootstrap_seed,
        )
        randomization = paired_group_randomization(
            design=design,
            metrics=args.metrics,
            n_randomizations=args.n_randomizations,
            seed=randomization_seed,
            batch_size=args.randomization_batch_size,
        )
        mcnemar = mcnemar_exact(y_true, pred_a, pred_b)
        comparison_id = (
            comparison_ids[pair_index - 1]
            if comparison_ids is not None
            else f"comparison_{pair_index:02d}"
        )

        comparison_results.append(
            {
                "comparison_id": comparison_id,
                "exp_a": exp_a,
                "exp_b": exp_b,
                "n_items": int(len(y_true)),
                "input_predictions": {
                    "exp_a_path": str(data_a.path.relative_to(ROOT_DIR)),
                    "exp_a_sha256": data_a.file_sha256,
                    "exp_b_path": str(data_b.path.relative_to(ROOT_DIR)),
                    "exp_b_sha256": data_b.file_sha256,
                },
                "group_design": design["diagnostics"],
                "bootstrap": bootstrap,
                "randomization": randomization,
                "mcnemar": mcnemar,
            }
        )

        for stratum in design["diagnostics"]["strata"]:
            group_diagnostic_rows.append(
                {
                    "comparison_id": comparison_id,
                    "exp_a": exp_a,
                    "exp_b": exp_b,
                    **stratum,
                }
            )

        for metric in args.metrics:
            effect = bootstrap["endpoints"][metric]
            test = randomization["endpoints"][metric]
            summary_rows.append(
                {
                    "analysis_scope": args.analysis_scope,
                    "confirmatory": args.analysis_scope == "confirmatory",
                    "subset": args.subset,
                    "comparison_id": comparison_id,
                    "exp_a": exp_a,
                    "exp_b": exp_b,
                    "metric": metric,
                    "endpoint_role": endpoint_role(metric),
                    "exp_a_value": effect["exp_a_value"],
                    "exp_b_value": effect["exp_b_value"],
                    "delta_a_minus_b": effect["delta_a_minus_b"],
                    "confidence_level": args.confidence_level,
                    "ci_low": effect["ci_low"],
                    "ci_high": effect["ci_high"],
                    "interval_method": bootstrap["interval"],
                    "bootstrap_unit": "cleaned_text_group",
                    "bootstrap_stratification": args.stratify,
                    "n_bootstrap": args.n_bootstrap,
                    "randomization_unit": "cleaned_text_group",
                    "n_randomizations": args.n_randomizations,
                    "extreme_randomizations": test["extreme_randomizations"],
                    "p_value_raw": test["p_value_raw"],
                    "p_value_resolution": test["p_value_resolution"],
                    "p_value_holm": None,
                    "holm_family": f"{args.analysis_scope}:{args.subset}:{metric}",
                    "holm_family_size": None,
                }
            )

    for metric in args.metrics:
        family_rows = [row for row in summary_rows if row["metric"] == metric]
        adjusted = holm_adjust([float(row["p_value_raw"]) for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted):
            row["p_value_holm"] = adjusted_p
            row["holm_family_size"] = len(family_rows)

    seed_rows, seed_summary_rows = analyze_seed_series(
        series_specs=seed_series_specs,
        metrics=args.metrics,
        mask=mask,
        get_prediction=get_prediction,
    )

    lineage_summary = {
        "path": str(lineage.path.relative_to(ROOT_DIR)),
        "sha256": lineage.file_sha256,
        "full_rows": int(len(mask)),
        "included_rows": int(mask.sum()),
        "excluded_rows": int(len(mask) - mask.sum()),
        "included_label_counts": {
            ID_TO_LABEL[label_id]: int(np.sum(lineage.label_id[mask] == label_id))
            for label_id in LABEL_IDS
        },
    }
    inputs = [
        {
            "experiment_id": item.experiment_id,
            "path": str(item.path.relative_to(ROOT_DIR)),
            "sha256": item.file_sha256,
            "rows": int(len(item.y_true)),
            "prediction_vector_sha256": prediction_vector_sha256(item.y_pred),
        }
        for item in sorted(prediction_cache.values(), key=lambda value: value.experiment_id)
    ]
    results = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "config": config,
        "lineage": lineage_summary,
        "inputs": inputs,
        "comparisons": comparison_results,
        "summary_rows": summary_rows,
        "seed_level": {
            "per_seed_rows": seed_rows,
            "summary_rows": seed_summary_rows,
        },
    }

    write_json(output_dir / "results.json", results)
    write_csv(
        output_dir / "endpoint_results.csv",
        summary_rows,
        [
            "analysis_scope",
            "confirmatory",
            "subset",
            "comparison_id",
            "exp_a",
            "exp_b",
            "metric",
            "endpoint_role",
            "exp_a_value",
            "exp_b_value",
            "delta_a_minus_b",
            "confidence_level",
            "ci_low",
            "ci_high",
            "interval_method",
            "bootstrap_unit",
            "bootstrap_stratification",
            "n_bootstrap",
            "randomization_unit",
            "n_randomizations",
            "extreme_randomizations",
            "p_value_raw",
            "p_value_resolution",
            "p_value_holm",
            "holm_family",
            "holm_family_size",
        ],
    )
    mcnemar_rows = [
        {
            "comparison_id": item["comparison_id"],
            "exp_a": item["exp_a"],
            "exp_b": item["exp_b"],
            **item["mcnemar"],
        }
        for item in comparison_results
    ]
    write_csv(
        output_dir / "mcnemar_results.csv",
        mcnemar_rows,
        [
            "comparison_id",
            "exp_a",
            "exp_b",
            "unit",
            "a_only_correct",
            "b_only_correct",
            "discordant_total",
            "p_value_exact",
            "p_value_exact_scientific",
            "log10_p_value_exact",
            "continuity_corrected_chi_square_diagnostic",
        ],
    )
    write_csv(
        output_dir / "group_strata.csv",
        group_diagnostic_rows,
        ["comparison_id", "exp_a", "exp_b", "stratum", "groups", "rows"],
    )
    if seed_rows:
        write_csv(
            output_dir / "seed_level_results.csv",
            seed_rows,
            [
                "series_id",
                "seed",
                "exp_a",
                "exp_b",
                "metric",
                "exp_a_value",
                "exp_b_value",
                "delta_a_minus_b",
            ],
        )
        write_csv(
            output_dir / "seed_level_summary.csv",
            seed_summary_rows,
            [
                "series_id",
                "description",
                "metric",
                "n_seed_pairs",
                "exp_a_mean",
                "exp_a_sample_sd",
                "exp_b_mean",
                "exp_b_sample_sd",
                "delta_mean",
                "delta_median",
                "delta_min",
                "delta_max",
                "delta_range",
                "exp_b_unique_prediction_vectors",
                "interpretation",
            ],
        )
    write_text(
        output_dir / "report.md",
        render_report(
            config=config,
            lineage_summary=lineage_summary,
            summary_rows=summary_rows,
            comparison_results=comparison_results,
            seed_summary_rows=seed_summary_rows,
        ),
    )

    if preserved_inputs is not None:
        observed = {relative: sha256_file(ROOT_DIR / relative) for relative in preserved_inputs}
        if observed != preserved_inputs:
            raise ValueError("Preserved statistical artifacts or sensitivity inputs changed during analysis.")
        write_json(output_dir / "preservation_check.json", {
            "status": "passed", "checked_files": len(observed),
            "all_recorded_inputs_unchanged": True,
            "original_weighted_analysis_unchanged": True,
            "training": False, "inference": False,
        })
    print(f"Wrote analysis artifacts to: {output_dir.relative_to(ROOT_DIR)}")
    return results


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run protocol-v2 paired statistical analysis on saved predictions"
    )
    parser.add_argument("--exp-a", nargs="+", default=None, dest="exp_a")
    parser.add_argument("--exp-b", nargs="+", default=None, dest="exp_b")
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=None,
        help="Immutable JSON contract for a named confirmatory or reviewer-sensitivity family",
    )
    parser.add_argument(
        "--analysis-scope",
        choices=["exploratory_continuity", "confirmatory", "reviewer_requested_sensitivity"],
        default="exploratory_continuity",
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=ALLOWED_METRICS)
    parser.add_argument("--subset", choices=["strict", "full", "strict_unique"], default="strict")
    parser.add_argument("--stratify", choices=["source_label", "label", "none"], default="source_label")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-randomizations", type=int, default=5000)
    parser.add_argument("--randomization-batch-size", type=int, default=32)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lineage-dir", type=Path, default=DEFAULT_LINEAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-seed-summary", action="store_true")
    args = parser.parse_args(argv)

    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")
    if args.n_randomizations < 100:
        parser.error("--n-randomizations must be at least 100")
    if args.randomization_batch_size < 1:
        parser.error("--randomization-batch-size must be positive")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be between 0 and 1")
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("--metrics cannot contain duplicates")
    if args.comparison_config is not None and (args.exp_a is not None or args.exp_b is not None):
        parser.error("--comparison-config cannot be combined with --exp-a/--exp-b")
    if args.comparison_config is not None and args.skip_seed_summary:
        parser.error("--skip-seed-summary cannot override an immutable comparison config")
    args.lineage_dir = args.lineage_dir.resolve()
    if args.comparison_config is not None:
        args.comparison_config = args.comparison_config.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT_DIR / "outputs" / "statistics" / f"significance_v2_{timestamp}"
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    command = shlex.join(["python", *sys.argv]) + "\n"
    write_text(output_dir / "command.txt", command)
    write_json(output_dir / "environment.json", environment_payload())
    write_json(
        output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "error": None,
        },
    )

    log_path = output_dir / "stdout.log"
    terminal_stdout = sys.stdout
    terminal_stderr = sys.stderr
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            with redirect_stdout(Tee(terminal_stdout, log_handle)), redirect_stderr(
                Tee(terminal_stderr, log_handle)
            ):
                run_analysis(args, output_dir)
        write_json(
            output_dir / "run_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "started_at_utc": json.loads(
                    (output_dir / "run_status.json").read_text(encoding="utf-8")
                )["started_at_utc"],
                "completed_at_utc": utc_now(),
                "error": None,
            },
        )
        write_json(
            output_dir / "artifact_manifest.json",
            build_manifest(output_dir, "complete"),
        )
    except Exception as error:
        with log_path.open("a", encoding="utf-8") as log_handle:
            traceback.print_exc(file=log_handle)
        current_status = json.loads(
            (output_dir / "run_status.json").read_text(encoding="utf-8")
        )
        write_json(
            output_dir / "run_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "started_at_utc": current_status["started_at_utc"],
                "completed_at_utc": utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        write_json(
            output_dir / "artifact_manifest.json",
            build_manifest(output_dir, "failed"),
        )
        raise

    print(f"Complete. Review: {output_dir / 'report.md'}")
    print(f"Machine-readable results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
