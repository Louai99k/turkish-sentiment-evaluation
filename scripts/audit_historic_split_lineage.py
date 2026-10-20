"""Audit historical Winvoker split lineage and leakage sensitivity.

The script reconstructs the exact split used by the completed experiments from
the pinned Hugging Face revision, verifies every internal prediction artifact
row-for-row, and recomputes metrics on prespecified filtered test subsets.

It never modifies the source dataset or existing experiment directories. All
outputs are written to a new audit directory for review before protocol v2 is
frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import clean_text


SCHEMA_VERSION = "historic-split-lineage-audit-v1"
DATASET_ID = "winvoker/turkish-sentiment-analysis-dataset"
PINNED_REVISION = "4bd9103ad5f24d44db806d40c94c8b9ec116ad05"
LABEL_TO_ID = {"Positive": 0, "Negative": 1, "Notr": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_IDS = [0, 1, 2]
PACKAGE_NAMES = (
    "datasets",
    "huggingface-hub",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_split_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    selected = frame[["source_row_index", "label", "__clean_text"]]
    for source_row_index, label, cleaned_text in selected.itertuples(
        index=False,
        name=None,
    ):
        payload = json.dumps(
            [
                int(source_row_index),
                str(label),
                str(cleaned_text),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def environment_payload() -> dict[str, Any]:
    packages = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hardware.update(
            {
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        hardware["torch_hardware_warning"] = repr(exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "packages": packages,
        "hardware": hardware,
    }


def build_historic_split(frame: pd.DataFrame, split_seed: int):
    train_validation, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=split_seed,
        stratify=frame["label_id"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.1,
        random_state=split_seed,
        stratify=train_validation["label_id"],
    )
    return {
        "train": train.copy(),
        "validation": validation.copy(),
        "test": test.copy(),
    }


def split_tables(splits: dict[str, pd.DataFrame]):
    summaries = []
    label_counts = []
    source_label_counts = []
    assignments = []

    for split_name, frame in splits.items():
        summaries.append(
            {
                "split": split_name,
                "rows": len(frame),
                "unique_raw_texts": frame["__raw_text"].nunique(dropna=False),
                "unique_cleaned_texts": frame["__clean_text"].nunique(dropna=False),
                "empty_cleaned_rows": frame["__clean_text"].eq("").sum(),
                "ordered_sha256": ordered_split_sha256(frame),
            }
        )

        labels = (
            frame.groupby(["label", "label_id"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        labels.insert(0, "split", split_name)
        label_counts.append(labels)

        sources = (
            frame.groupby(["dataset", "label"], dropna=False)
            .size()
            .reset_index(name="count")
            .rename(columns={"dataset": "source"})
        )
        sources.insert(0, "split", split_name)
        source_label_counts.append(sources)

        selected = frame[["source_row_index", "label", "label_id", "dataset"]].copy()
        selected.insert(0, "order_in_split", np.arange(len(selected), dtype=np.int64))
        selected.insert(0, "split", split_name)
        assignments.append(selected)

    return (
        pd.DataFrame(summaries),
        pd.concat(label_counts, ignore_index=True),
        pd.concat(source_label_counts, ignore_index=True),
        pd.concat(assignments, ignore_index=True),
    )


def build_test_mask(
    full_frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    train_clean = set(splits["train"]["__clean_text"])
    validation_clean = set(splits["validation"]["__clean_text"])
    test = splits["test"].copy()

    label_cardinality = full_frame.groupby("__clean_text", dropna=False)["label"].nunique()
    conflicting_clean = set(label_cardinality[label_cardinality > 1].index)

    test["is_empty_after_cleaning"] = test["__clean_text"].eq("")
    test["overlaps_train_cleaned"] = test["__clean_text"].isin(train_clean)
    test["overlaps_validation_cleaned"] = test["__clean_text"].isin(validation_clean)
    test["has_global_cleaned_label_conflict"] = test["__clean_text"].isin(
        conflicting_clean
    )
    test["is_duplicate_within_test"] = test["__clean_text"].duplicated(keep=False)
    test["strict_included"] = ~(
        test["is_empty_after_cleaning"]
        | test["overlaps_train_cleaned"]
        | test["overlaps_validation_cleaned"]
        | test["has_global_cleaned_label_conflict"]
    )
    test["strict_unique_included"] = test["strict_included"] & ~test[
        "__clean_text"
    ].duplicated(keep="first")

    mask = test[
        [
            "source_row_index",
            "label",
            "label_id",
            "dataset",
            "is_empty_after_cleaning",
            "overlaps_train_cleaned",
            "overlaps_validation_cleaned",
            "has_global_cleaned_label_conflict",
            "is_duplicate_within_test",
            "strict_included",
            "strict_unique_included",
        ]
    ].copy()
    mask.insert(0, "test_position", np.arange(len(mask), dtype=np.int64))
    mask["raw_text_sha256"] = test["__raw_text"].map(text_sha256).to_numpy()
    mask["cleaned_text_sha256"] = test["__clean_text"].map(text_sha256).to_numpy()

    summary = {
        "test_rows": len(test),
        "empty_after_cleaning": int(test["is_empty_after_cleaning"].sum()),
        "overlaps_train_cleaned": int(test["overlaps_train_cleaned"].sum()),
        "overlaps_validation_cleaned": int(
            test["overlaps_validation_cleaned"].sum()
        ),
        "global_conflicting_label_rows": int(
            test["has_global_cleaned_label_conflict"].sum()
        ),
        "rows_in_test_duplicate_groups": int(test["is_duplicate_within_test"].sum()),
        "excluded_by_any_strict_rule": int((~test["strict_included"]).sum()),
        "strict_included_rows": int(test["strict_included"].sum()),
        "strict_unique_included_rows": int(test["strict_unique_included"].sum()),
    }

    conflict_examples = full_frame[
        full_frame["__clean_text"].isin(conflicting_clean)
    ].copy()
    conflict_examples = (
        conflict_examples.groupby(["__clean_text", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["__clean_text", "label"])
        .head(300)
        .rename(columns={"__clean_text": "cleaned_text"})
    )
    return mask, summary, conflict_examples


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    averaging_labels: list[int] | None = None,
) -> dict[str, Any]:
    if averaging_labels is None:
        averaging_labels = LABEL_IDS
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=averaging_labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=averaging_labels,
        average="weighted",
        zero_division=0,
    )
    per_class_values = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        average=None,
        zero_division=0,
    )
    per_class = {}
    for position, label_id in enumerate(LABEL_IDS):
        per_class[ID_TO_LABEL[label_id]] = {
            "precision": per_class_values[0][position],
            "recall": per_class_values[1][position],
            "f1": per_class_values[2][position],
            "support": per_class_values[3][position],
        }
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
        "negative_f1": per_class["Negative"]["f1"],
        "notr_prediction_rate": float(np.mean(y_pred == LABEL_TO_ID["Notr"])),
        "averaging_labels": [ID_TO_LABEL[label_id] for label_id in averaging_labels],
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABEL_IDS,
        ).tolist(),
    }


def flat_metric_row(experiment_id: str, subset: str, metrics: dict[str, Any]):
    row = {
        "experiment_id": experiment_id,
        "subset": subset,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "negative_f1": metrics["negative_f1"],
        "notr_prediction_rate": metrics["notr_prediction_rate"],
    }
    for label in ("Positive", "Negative", "Notr"):
        prefix = label.lower()
        row[f"{prefix}_precision"] = metrics["per_class"][label]["precision"]
        row[f"{prefix}_recall"] = metrics["per_class"][label]["recall"]
        row[f"{prefix}_f1"] = metrics["per_class"][label]["f1"]
        row[f"{prefix}_support"] = metrics["per_class"][label]["support"]
    return row


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_labels(frame: pd.DataFrame, id_column: str, name_column: str) -> pd.Series:
    if id_column in frame.columns:
        values = pd.to_numeric(frame[id_column], errors="coerce")
    elif name_column in frame.columns:
        values = frame[name_column].map(LABEL_TO_ID)
    else:
        raise ValueError(f"Prediction file lacks {id_column!r} and {name_column!r}")
    if values.isna().any():
        raise ValueError(f"Prediction file contains unknown values in {id_column}/{name_column}")
    return values.astype(int)


def audit_predictions(
    experiments_dir: Path,
    expected_test: pd.DataFrame,
    test_mask: pd.DataFrame,
):
    lineage_rows = []
    mismatch_rows = []
    metric_rows = []
    source_metric_rows = []
    metric_payload = {}

    expected_text = expected_test["__clean_text"].reset_index(drop=True)
    expected_labels = expected_test["label_id"].reset_index(drop=True).astype(int)
    expected_sources = expected_test["dataset"].reset_index(drop=True).astype(str)
    strict_mask = test_mask["strict_included"].to_numpy(dtype=bool)
    strict_unique_mask = test_mask["strict_unique_included"].to_numpy(dtype=bool)

    prediction_paths = sorted(experiments_dir.glob("*/predictions.csv"))
    print(f"Found {len(prediction_paths)} prediction artifacts")
    for prediction_path in prediction_paths:
        experiment_dir = prediction_path.parent
        experiment_id = experiment_dir.name
        config = load_json(experiment_dir / "config.json")
        stored_metrics = load_json(experiment_dir / "metrics.json")

        if experiment_id.startswith("x1_cross_"):
            lineage_rows.append(
                {
                    "experiment_id": experiment_id,
                    "scope": "external_excluded",
                    "model": config.get("model"),
                    "seed": config.get("seed"),
                    "row_count": None,
                    "row_count_match": False,
                    "text_mismatch_rows": None,
                    "true_label_mismatch_rows": None,
                    "aligned": False,
                    "superseded": False,
                    "stored_weighted_f1": stored_metrics.get("weighted_f1"),
                    "recomputed_weighted_f1": None,
                    "weighted_f1_delta": None,
                    "stored_metrics_match": None,
                }
            )
            continue

        print(f"Checking prediction lineage: {experiment_id}")
        predictions = pd.read_csv(prediction_path, keep_default_na=False)
        row_count_match = len(predictions) == len(expected_test)
        text_mismatches = None
        label_mismatches = None
        y_true = None
        y_pred = None

        try:
            y_true = numeric_labels(predictions, "true_id", "true_label")
            y_pred = numeric_labels(predictions, "predicted_id", "predicted_label")
            if row_count_match:
                prediction_text = predictions["text"].astype(str).reset_index(drop=True)
                text_difference = prediction_text.ne(expected_text)
                label_difference = y_true.reset_index(drop=True).ne(expected_labels)
                text_mismatches = int(text_difference.sum())
                label_mismatches = int(label_difference.sum())

                mismatch_positions = np.flatnonzero(
                    text_difference.to_numpy() | label_difference.to_numpy()
                )[:25]
                for position in mismatch_positions:
                    mismatch_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "test_position": int(position),
                            "expected_source_row_index": int(
                                expected_test.iloc[position]["source_row_index"]
                            ),
                            "expected_text": expected_text.iloc[position],
                            "prediction_text": prediction_text.iloc[position],
                            "expected_label_id": int(expected_labels.iloc[position]),
                            "prediction_true_id": int(y_true.iloc[position]),
                        }
                    )
        except Exception as exc:
            mismatch_rows.append(
                {
                    "experiment_id": experiment_id,
                    "test_position": None,
                    "expected_source_row_index": None,
                    "expected_text": None,
                    "prediction_text": None,
                    "expected_label_id": None,
                    "prediction_true_id": None,
                    "error": repr(exc),
                }
            )

        aligned = (
            row_count_match
            and text_mismatches == 0
            and label_mismatches == 0
            and y_true is not None
            and y_pred is not None
        )
        superseded = experiment_id == "s1_berturk_berturk_winvoker_seed42_weighted"
        recomputed = {
            "accuracy": None,
            "weighted_f1": None,
            "macro_f1": None,
            "negative_f1": None,
        }
        deltas = {metric: None for metric in recomputed}
        stored_metrics_match = None

        if aligned:
            y_true_array = y_true.to_numpy(dtype=int)
            y_pred_array = y_pred.to_numpy(dtype=int)
            subset_masks = {
                "full_historic_test": np.ones(len(expected_test), dtype=bool),
                "strict_filtered_test": strict_mask,
                "strict_unique_text_test": strict_unique_mask,
                "polarity_only_full": np.isin(y_true_array, [0, 1]),
                "polarity_only_strict": strict_mask & np.isin(y_true_array, [0, 1]),
            }
            run_metrics = {}
            for subset_name, subset_mask in subset_masks.items():
                averaging_labels = (
                    [LABEL_TO_ID["Positive"], LABEL_TO_ID["Negative"]]
                    if subset_name.startswith("polarity_only_")
                    else LABEL_IDS
                )
                metrics = classification_metrics(
                    y_true_array[subset_mask],
                    y_pred_array[subset_mask],
                    averaging_labels=averaging_labels,
                )
                run_metrics[subset_name] = metrics
                metric_rows.append(flat_metric_row(experiment_id, subset_name, metrics))

            for subset_name, base_mask in {
                "full_historic_test": subset_masks["full_historic_test"],
                "strict_filtered_test": strict_mask,
            }.items():
                for source in sorted(expected_sources.unique()):
                    source_mask = base_mask & expected_sources.eq(source).to_numpy()
                    if not source_mask.any():
                        continue
                    metrics = classification_metrics(
                        y_true_array[source_mask],
                        y_pred_array[source_mask],
                    )
                    row = flat_metric_row(experiment_id, subset_name, metrics)
                    row["source"] = source
                    source_metric_rows.append(row)

            metric_payload[experiment_id] = run_metrics
            full_metrics = run_metrics["full_historic_test"]
            for metric in recomputed:
                recomputed[metric] = full_metrics[metric]
                if stored_metrics.get(metric) is not None:
                    deltas[metric] = recomputed[metric] - float(stored_metrics[metric])
            available_deltas = [value for value in deltas.values() if value is not None]
            stored_metrics_match = bool(available_deltas) and all(
                abs(value) <= 1e-12 for value in available_deltas
            )

        lineage_rows.append(
            {
                "experiment_id": experiment_id,
                "scope": "internal_winvoker",
                "model": config.get("model"),
                "seed": config.get("seed"),
                "row_count": len(predictions),
                "row_count_match": row_count_match,
                "text_mismatch_rows": text_mismatches,
                "true_label_mismatch_rows": label_mismatches,
                "aligned": aligned,
                "superseded": superseded,
                "stored_accuracy": stored_metrics.get("accuracy"),
                "recomputed_accuracy": recomputed["accuracy"],
                "accuracy_delta": deltas["accuracy"],
                "stored_weighted_f1": stored_metrics.get("weighted_f1"),
                "recomputed_weighted_f1": recomputed["weighted_f1"],
                "weighted_f1_delta": deltas["weighted_f1"],
                "stored_macro_f1": stored_metrics.get("macro_f1"),
                "recomputed_macro_f1": recomputed["macro_f1"],
                "macro_f1_delta": deltas["macro_f1"],
                "stored_negative_f1": stored_metrics.get("negative_f1"),
                "recomputed_negative_f1": recomputed["negative_f1"],
                "negative_f1_delta": deltas["negative_f1"],
                "stored_metrics_match": stored_metrics_match,
            }
        )

    return (
        pd.DataFrame(lineage_rows),
        pd.DataFrame(mismatch_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(source_metric_rows),
        metric_payload,
    )


def markdown_notes(audit: dict[str, Any], lineage: pd.DataFrame) -> str:
    internal = lineage[lineage["scope"].eq("internal_winvoker")]
    aligned = int(internal["aligned"].sum()) if not internal.empty else 0
    lines = [
        "# Historical Split Lineage and Leakage Sensitivity Audit",
        "",
        f"- Dataset: `{audit['dataset']['id']}`",
        f"- Immutable revision: `{audit['dataset']['revision']}`",
        f"- Historical split seed: `{audit['split']['seed']}`",
        f"- Internal prediction artifacts aligned: `{aligned}/{len(internal)}`",
        f"- Historical test rows: `{audit['test_filter']['test_rows']}`",
        f"- Strict filtered test rows: `{audit['test_filter']['strict_included_rows']}`",
        f"- Strict unique-text test rows: `{audit['test_filter']['strict_unique_included_rows']}`",
        "",
        "## Strict Filter",
        "",
        "A test row is excluded if its cleaned text is empty, occurs in historical training or validation data, or belongs to a cleaned-text group with conflicting labels anywhere in the 440,679-row historical corpus.",
        "",
        "The unique-text sensitivity subset additionally retains only the first test occurrence of each cleaned text. It is diagnostic and does not silently replace the prespecified main metric.",
        "",
        "## Interpretation Boundary",
        "",
        "- Alignment validates row lineage; it does not add missing dataset/model revision metadata to legacy run configurations.",
        "- Small metric changes would support retaining old runs with disclosure, but do not remove the source-label confound.",
        "- Large changes or failed alignment require protocol-v2 reruns.",
        "- Near-duplicate semantic leakage is outside this exact-text audit.",
        "- Source-specific metrics are diagnostic because several sources do not contain all three labels.",
    ]
    return "\n".join(lines) + "\n"


def build_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "artifacts": artifacts,
    }


def run_audit(args, output_dir: Path) -> dict[str, Any]:
    print(f"Loading pinned dataset revision: {args.revision}")
    dataset = load_dataset(args.dataset_name, revision=args.revision)
    if "train" not in dataset:
        raise ValueError("Pinned dataset has no publisher 'train' split")

    publisher_train = dataset["train"].select_columns(
        ["text", "label", "dataset"]
    ).to_pandas()
    publisher_train.insert(
        0,
        "source_row_index",
        np.arange(len(publisher_train), dtype=np.int64),
    )
    publisher_train["__raw_text"] = publisher_train["text"].fillna("").astype(str)
    publisher_train["__clean_text"] = publisher_train["__raw_text"].map(clean_text)
    publisher_train["label_id"] = publisher_train["label"].map(LABEL_TO_ID)
    if publisher_train["label_id"].isna().any():
        unknown = sorted(publisher_train.loc[publisher_train["label_id"].isna(), "label"].unique())
        raise ValueError(f"Unknown publisher labels: {unknown}")
    publisher_train["label_id"] = publisher_train["label_id"].astype(int)

    print("Reconstructing historical 72/8/20 split")
    splits = build_historic_split(publisher_train, args.split_seed)
    split_summary, label_counts, source_label_counts, assignments = split_tables(splits)
    test_mask, filter_summary, conflict_examples = build_test_mask(
        publisher_train,
        splits,
    )

    print("Auditing saved prediction artifacts")
    (
        lineage,
        mismatches,
        sensitivity_metrics,
        source_metrics,
        metric_payload,
    ) = audit_predictions(args.experiments_dir, splits["test"], test_mask)

    internal = lineage[lineage["scope"].eq("internal_winvoker")]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "dataset": {
            "id": args.dataset_name,
            "revision": args.revision,
            "source_split_used_historically": "train",
            "publisher_train_rows": len(publisher_train),
        },
        "split": {
            "seed": args.split_seed,
            "method": "sklearn train_test_split; stratified 80/20 then train portion 90/10",
            "summary": split_summary.to_dict(orient="records"),
        },
        "test_filter": filter_summary,
        "prediction_lineage": {
            "discovered_prediction_artifacts": len(lineage),
            "internal_prediction_artifacts": len(internal),
            "aligned_internal_artifacts": int(internal["aligned"].sum()),
            "unaligned_internal_artifacts": int((~internal["aligned"]).sum()),
            "external_artifacts_excluded": int(
                lineage["scope"].eq("external_excluded").sum()
            ),
        },
        "eligibility_rule": {
            "retain_existing_runs_if": [
                "row_count, cleaned text order, and true-label order all align",
                "recomputed full metrics match stored metrics within numeric tolerance",
                "strict-filter sensitivity does not materially reverse conclusions",
            ],
            "remaining_limitations": [
                "legacy configs do not pin dataset/model revisions",
                "source and label are strongly confounded",
                "near-duplicate semantic leakage is not measured here",
            ],
        },
    }

    atomic_write_json(output_dir / "audit.json", audit)
    atomic_write_json(output_dir / "sensitivity_metrics.json", metric_payload)
    atomic_write_csv(output_dir / "split_summary.csv", split_summary)
    atomic_write_csv(output_dir / "split_label_counts.csv", label_counts)
    atomic_write_csv(
        output_dir / "split_source_label_counts.csv",
        source_label_counts,
    )
    atomic_write_csv(output_dir / "split_assignments.csv", assignments)
    atomic_write_csv(output_dir / "test_inclusion_mask.csv", test_mask)
    atomic_write_csv(
        output_dir / "global_conflicting_cleaned_examples.csv",
        conflict_examples,
    )
    atomic_write_csv(output_dir / "prediction_lineage.csv", lineage)
    atomic_write_csv(
        output_dir / "prediction_mismatch_examples.csv",
        mismatches,
    )
    atomic_write_csv(
        output_dir / "sensitivity_metrics.csv",
        sensitivity_metrics,
    )
    atomic_write_csv(output_dir / "source_slice_metrics.csv", source_metrics)
    atomic_write_text(output_dir / "notes.md", markdown_notes(audit, lineage))
    return audit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit historical split lineage and exact-text leakage sensitivity"
    )
    parser.add_argument("--dataset-name", default=DATASET_ID)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "audits" / "historic_split_lineage_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.experiments_dir.is_absolute():
        args.experiments_dir = ROOT_DIR / args.experiments_dir
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT_DIR / args.output_dir

    if args.revision != PINNED_REVISION:
        raise SystemExit(
            "This lineage audit must use the audited immutable revision "
            f"{PINNED_REVISION}; received {args.revision}."
        )
    if not args.experiments_dir.is_dir():
        raise SystemExit(f"Experiments directory not found: {args.experiments_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose another directory or pass --overwrite intentionally."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = shlex.join([sys.executable, *sys.argv])
    atomic_write_text(args.output_dir / "command.txt", command + "\n")
    atomic_write_json(args.output_dir / "environment.json", environment_payload())
    started_at = utc_now()
    atomic_write_json(
        args.output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": started_at,
            "completed_at_utc": None,
            "error": None,
        },
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with (args.output_dir / "stdout.log").open("w", encoding="utf-8") as log_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)
        try:
            print(f"Historical split lineage audit started: {started_at}")
            print(f"Output directory: {args.output_dir}")
            audit = run_audit(args, args.output_dir)
            completed_at = utc_now()
            atomic_write_json(
                args.output_dir / "config.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": args.dataset_name,
                    "revision": args.revision,
                    "split_seed": args.split_seed,
                    "experiments_dir": str(args.experiments_dir.relative_to(ROOT_DIR)),
                    "output_dir": str(args.output_dir.relative_to(ROOT_DIR)),
                },
            )
            atomic_write_json(
                args.output_dir / "run_status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                    "error": None,
                },
            )
            print(f"Historical split lineage audit complete: {completed_at}")
            print(
                "Aligned internal prediction artifacts: "
                f"{audit['prediction_lineage']['aligned_internal_artifacts']}/"
                f"{audit['prediction_lineage']['internal_prediction_artifacts']}"
            )
        except Exception as exc:
            completed_at = utc_now()
            error_payload = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(
                args.output_dir / "run_status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                    "error": error_payload,
                },
            )
            print(error_payload["traceback"], file=sys.stderr)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    atomic_write_json(
        args.output_dir / "artifact_manifest.json",
        build_manifest(args.output_dir),
    )
    print(f"Artifact manifest: {args.output_dir / 'artifact_manifest.json'}")


if __name__ == "__main__":
    main()
