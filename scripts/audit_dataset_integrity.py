"""Audit dataset revision, distributions, duplicates, and split integrity.

This script is intentionally read-only with respect to source datasets. It writes
all audit results under one output directory so another reviewer can inspect the
run without relying on terminal output.

Intended execution environment: the project's Docker container.
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
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import clean_text


SCHEMA_VERSION = "dataset-integrity-audit-v1"
DEFAULT_DATASET = "winvoker/turkish-sentiment-analysis-dataset"
TEXT_CANDIDATES = ("text", "sentence", "content", "tweet", "review")
LABEL_CANDIDATES = ("label", "sentiment", "polarity", "target")
SOURCE_CANDIDATES = ("dataset", "source", "domain", "category")
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
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
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_column(columns, explicit, candidates, role, required=True):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Requested {role} column {explicit!r} not in {list(columns)}")
        return explicit
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Could not detect {role} column from {list(columns)}")
    return None


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
    except Exception as exc:  # Environment reporting must not invalidate the audit.
        hardware["torch_hardware_warning"] = repr(exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "packages": packages,
        "hardware": hardware,
    }


def resolve_dataset_revision(dataset_name: str, requested_revision: str | None):
    reference = requested_revision or "main"
    # Hub metadata requests can be rejected temporarily when the public request
    # queue is full. Retry only HTTP 429 responses; all other failures retain
    # their original fail-fast behavior.
    import time

    max_attempts = 5
    initial_delay_seconds = 10.0
    for attempt in range(1, max_attempts + 1):
        try:
            info = HfApi().dataset_info(
                repo_id=dataset_name,
                revision=reference,
            )
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            rate_limited = (
                status_code == 429 or "429 Too Many Requests" in str(exc)
            )

            if not rate_limited or attempt == max_attempts:
                raise

            delay = initial_delay_seconds * (2 ** (attempt - 1))
            print(
                "Hugging Face returned HTTP 429 while resolving the dataset "
                f"revision (attempt {attempt}/{max_attempts}). "
                f"Retrying in {delay:.0f} seconds...",
                flush=True,
            )
            time.sleep(delay)
    return info.sha, {
        "id": info.id,
        "sha": info.sha,
        "requested_revision": requested_revision,
        "resolved_reference": reference,
        "last_modified": str(info.last_modified) if info.last_modified else None,
        "private": info.private,
        "gated": info.gated,
        "card_data": str(info.card_data) if info.card_data is not None else None,
    }


def split_to_frame(split_dataset, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column is not None]
    frame = split_dataset.select_columns(available).to_pandas()
    frame.insert(0, "source_row_index", np.arange(len(frame), dtype=np.int64))
    return frame


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def count_table(frame, split_name, column, value_name):
    counts = frame[column].fillna("<NULL>").astype(str).value_counts(dropna=False)
    result = counts.rename_axis(value_name).reset_index(name="count")
    result.insert(0, "split", split_name)
    result["proportion"] = result["count"] / max(len(frame), 1)
    return result


def duplicate_summary(frame, split_name, text_column, label_column):
    text = normalized_text(frame[text_column])
    work = pd.DataFrame({"text": text, "label": frame[label_column].astype(str)})
    label_cardinality = work.groupby("text", dropna=False)["label"].nunique(dropna=False)
    conflicting_texts = set(label_cardinality[label_cardinality > 1].index)
    duplicated_any = text.duplicated(keep=False)

    summary = {
        "split": split_name,
        "rows": len(frame),
        "unique_texts": int(text.nunique(dropna=False)),
        "duplicate_excess_rows": int(len(frame) - text.nunique(dropna=False)),
        "rows_in_duplicate_groups": int(duplicated_any.sum()),
        "duplicate_text_groups": int(text[text.duplicated(keep=False)].nunique()),
        "conflicting_label_text_groups": len(conflicting_texts),
        "rows_in_conflicting_label_groups": int(text.isin(conflicting_texts).sum()),
        "empty_text_rows": int(text.str.strip().eq("").sum()),
    }

    examples = work[work["text"].isin(conflicting_texts)].copy()
    if not examples.empty:
        examples = (
            examples.groupby(["text", "label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["text", "label"])
            .head(200)
        )
        examples.insert(0, "split", split_name)
    return summary, examples


def cramers_v(contingency: pd.DataFrame) -> dict[str, Any] | None:
    observed = contingency.to_numpy(dtype=float)
    if observed.size == 0 or observed.sum() == 0:
        return None
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / observed.sum()
    valid = expected > 0
    chi_square = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
    value = float(np.sqrt(chi_square / denominator)) if denominator > 0 else None
    return {
        "chi_square": chi_square,
        "cramers_v": value,
        "n": int(observed.sum()),
        "rows": observed.shape[0],
        "columns": observed.shape[1],
    }


def cross_split_overlap(frames, text_column, representation):
    split_names = list(frames)
    rows = []
    samples = []
    unique_sets = {
        name: set(normalized_text(frame[text_column])) for name, frame in frames.items()
    }
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = unique_sets[left_name] & unique_sets[right_name]
            rows.append(
                {
                    "representation": representation,
                    "left_split": left_name,
                    "right_split": right_name,
                    "left_unique_texts": len(unique_sets[left_name]),
                    "right_unique_texts": len(unique_sets[right_name]),
                    "overlapping_unique_texts": len(overlap),
                }
            )
            for text in sorted(overlap)[:50]:
                samples.append(
                    {
                        "representation": representation,
                        "left_split": left_name,
                        "right_split": right_name,
                        "text": text,
                    }
                )
    return rows, samples


def build_custom_split(train_frame, label_column, split_seed):
    train_val, test = train_test_split(
        train_frame,
        test_size=0.2,
        random_state=split_seed,
        stratify=train_frame[label_column],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.1,
        random_state=split_seed,
        stratify=train_val[label_column],
    )
    return {
        "custom_train": train.copy(),
        "custom_validation": validation.copy(),
        "custom_test": test.copy(),
    }


def markdown_notes(audit):
    lines = [
        "# Dataset Integrity Audit",
        "",
        f"- Dataset: `{audit['dataset']['id']}`",
        f"- Resolved revision: `{audit['dataset']['sha']}`",
        f"- Completed: `{audit['completed_at_utc']}`",
        f"- Text column: `{audit['columns']['text']}`",
        f"- Label column: `{audit['columns']['label']}`",
        f"- Source/domain column: `{audit['columns']['source']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Unique raw texts | Unique cleaned texts | Empty after cleaning |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for row in audit["split_summary"]:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['unique_raw_texts']} | "
            f"{row['unique_cleaned_texts']} | {row['empty_after_cleaning']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This audit describes data integrity and composition; it does not validate model performance.",
            "- Exact and cleaned-text overlap should be reviewed before a split is declared leakage-safe.",
            "- Source/domain association with labels can reveal confounding but does not alone prove a model used that shortcut.",
            "- The resolved immutable revision must be used by future paper-eligible runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name.endswith(".tmp") or path.name == "artifact_manifest.json":
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
    print(f"Resolving dataset revision: {args.dataset_name} ({args.revision or 'main'})")
    resolved_revision, dataset_metadata = resolve_dataset_revision(
        args.dataset_name, args.revision
    )
    print(f"Resolved immutable revision: {resolved_revision}")
    print("Loading dataset at resolved revision...")
    dataset = load_dataset(args.dataset_name, revision=resolved_revision)
    if not dataset:
        raise ValueError("Dataset contains no splits")

    first_split = next(iter(dataset.values()))
    columns = list(first_split.column_names)
    text_column = resolve_column(columns, args.text_column, TEXT_CANDIDATES, "text")
    label_column = resolve_column(columns, args.label_column, LABEL_CANDIDATES, "label")
    source_column = resolve_column(
        columns, args.source_column, SOURCE_CANDIDATES, "source/domain", required=False
    )
    selected_columns = [text_column, label_column]
    if source_column:
        selected_columns.append(source_column)

    frames = {}
    fingerprints = {}
    label_tables = []
    source_tables = []
    source_label_tables = []
    raw_duplicate_rows = []
    clean_duplicate_rows = []
    conflicting_examples = []
    split_summary = []

    for split_name, split_dataset in dataset.items():
        print(f"Auditing publisher split: {split_name} ({len(split_dataset)} rows)")
        frame = split_to_frame(split_dataset, selected_columns)
        frame["__raw_text"] = normalized_text(frame[text_column])
        frame["__clean_text"] = frame["__raw_text"].map(clean_text)
        frames[split_name] = frame
        fingerprints[split_name] = getattr(split_dataset, "_fingerprint", None)

        label_tables.append(count_table(frame, split_name, label_column, "label"))
        if source_column:
            source_tables.append(count_table(frame, split_name, source_column, "source"))
            source_label = (
                frame.groupby([source_column, label_column], dropna=False)
                .size()
                .reset_index(name="count")
                .rename(columns={source_column: "source", label_column: "label"})
            )
            source_label.insert(0, "split", split_name)
            source_label_tables.append(source_label)

        raw_summary, raw_examples = duplicate_summary(
            frame, split_name, "__raw_text", label_column
        )
        raw_summary["representation"] = "raw"
        raw_duplicate_rows.append(raw_summary)
        if not raw_examples.empty:
            raw_examples.insert(1, "representation", "raw")
            conflicting_examples.append(raw_examples)

        clean_summary, clean_examples = duplicate_summary(
            frame, split_name, "__clean_text", label_column
        )
        clean_summary["representation"] = "cleaned"
        clean_duplicate_rows.append(clean_summary)
        if not clean_examples.empty:
            clean_examples.insert(1, "representation", "cleaned")
            conflicting_examples.append(clean_examples)

        split_summary.append(
            {
                "split": split_name,
                "rows": len(frame),
                "unique_raw_texts": raw_summary["unique_texts"],
                "unique_cleaned_texts": clean_summary["unique_texts"],
                "empty_before_cleaning": raw_summary["empty_text_rows"],
                "empty_after_cleaning": clean_summary["empty_text_rows"],
                "fingerprint": fingerprints[split_name],
            }
        )

    overlap_rows = []
    overlap_samples = []
    raw_rows, raw_samples = cross_split_overlap(frames, "__raw_text", "raw")
    clean_rows, clean_samples = cross_split_overlap(frames, "__clean_text", "cleaned")
    overlap_rows.extend(raw_rows)
    overlap_rows.extend(clean_rows)
    overlap_samples.extend(raw_samples)
    overlap_samples.extend(clean_samples)

    custom_split_summary = []
    custom_overlap_rows = []
    custom_overlap_samples = []
    if args.simulate_historic_split:
        if "train" not in frames:
            raise ValueError("Cannot simulate historic split because publisher split 'train' is absent")
        print("Simulating historic 72/8/20 stratified split from publisher train split...")
        custom_frames = build_custom_split(frames["train"], label_column, args.split_seed)
        for split_name, frame in custom_frames.items():
            raw_summary, _ = duplicate_summary(frame, split_name, "__raw_text", label_column)
            clean_summary, _ = duplicate_summary(frame, split_name, "__clean_text", label_column)
            custom_split_summary.append(
                {
                    "split": split_name,
                    "rows": len(frame),
                    "unique_raw_texts": raw_summary["unique_texts"],
                    "unique_cleaned_texts": clean_summary["unique_texts"],
                    "empty_after_cleaning": clean_summary["empty_text_rows"],
                }
            )
        raw_rows, raw_samples = cross_split_overlap(
            custom_frames, "__raw_text", "raw"
        )
        clean_rows, clean_samples = cross_split_overlap(
            custom_frames, "__clean_text", "cleaned"
        )
        custom_overlap_rows.extend(raw_rows)
        custom_overlap_rows.extend(clean_rows)
        custom_overlap_samples.extend(raw_samples)
        custom_overlap_samples.extend(clean_samples)

    label_counts = pd.concat(label_tables, ignore_index=True)
    source_counts = (
        pd.concat(source_tables, ignore_index=True) if source_tables else pd.DataFrame()
    )
    source_label_counts = (
        pd.concat(source_label_tables, ignore_index=True)
        if source_label_tables
        else pd.DataFrame()
    )
    duplicate_frame = pd.DataFrame(raw_duplicate_rows + clean_duplicate_rows)
    conflict_frame = (
        pd.concat(conflicting_examples, ignore_index=True)
        if conflicting_examples
        else pd.DataFrame(columns=["split", "representation", "text", "label", "count"])
    )

    association = None
    if source_column:
        combined = pd.concat(frames.values(), ignore_index=True)
        contingency = pd.crosstab(combined[source_column], combined[label_column])
        association = cramers_v(contingency)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "dataset": dataset_metadata,
        "columns": {
            "available": columns,
            "text": text_column,
            "label": label_column,
            "source": source_column,
        },
        "publisher_split_fingerprints": fingerprints,
        "split_summary": split_summary,
        "duplicate_summary": duplicate_frame.to_dict(orient="records"),
        "publisher_cross_split_overlap": overlap_rows,
        "source_label_association": association,
        "historic_split_simulation": {
            "enabled": args.simulate_historic_split,
            "split_seed": args.split_seed if args.simulate_historic_split else None,
            "summary": custom_split_summary,
            "cross_split_overlap": custom_overlap_rows,
        },
    }

    atomic_write_json(output_dir / "audit.json", audit)
    atomic_write_csv(output_dir / "split_summary.csv", pd.DataFrame(split_summary))
    atomic_write_csv(output_dir / "label_counts.csv", label_counts)
    atomic_write_csv(output_dir / "duplicate_summary.csv", duplicate_frame)
    atomic_write_csv(output_dir / "conflicting_label_examples.csv", conflict_frame)
    atomic_write_csv(output_dir / "cross_split_overlap.csv", pd.DataFrame(overlap_rows))
    atomic_write_csv(
        output_dir / "cross_split_overlap_examples.csv", pd.DataFrame(overlap_samples)
    )
    if not source_counts.empty:
        atomic_write_csv(output_dir / "source_counts.csv", source_counts)
    if not source_label_counts.empty:
        atomic_write_csv(output_dir / "source_label_counts.csv", source_label_counts)
    if args.simulate_historic_split:
        atomic_write_csv(
            output_dir / "historic_split_summary.csv",
            pd.DataFrame(custom_split_summary),
        )
        atomic_write_csv(
            output_dir / "historic_split_overlap.csv",
            pd.DataFrame(custom_overlap_rows),
        )
        atomic_write_csv(
            output_dir / "historic_split_overlap_examples.csv",
            pd.DataFrame(custom_overlap_samples),
        )
    atomic_write_text(output_dir / "notes.md", markdown_notes(audit))
    return audit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a structured integrity audit for a Hugging Face text dataset"
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--revision",
        default=None,
        help="Revision/commit to resolve. If omitted, main is resolved to an immutable SHA.",
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--simulate-historic-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/audits/winvoker_current_main",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = shlex.join([sys.executable, *sys.argv])
    atomic_write_text(output_dir / "command.txt", command + "\n")
    atomic_write_json(
        output_dir / "environment.json",
        environment_payload(),
    )
    atomic_write_json(
        output_dir / "run_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "error": None,
        },
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_path = output_dir / "stdout.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)
        started_at = utc_now()
        try:
            print(f"Dataset integrity audit started: {started_at}")
            print(f"Output directory: {output_dir}")
            audit = run_audit(args, output_dir)
            completed_at = utc_now()
            atomic_write_json(
                output_dir / "config.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": args.dataset_name,
                    "requested_revision": args.revision,
                    "resolved_revision": audit["dataset"]["sha"],
                    "text_column": audit["columns"]["text"],
                    "label_column": audit["columns"]["label"],
                    "source_column": audit["columns"]["source"],
                    "split_seed": args.split_seed,
                    "simulate_historic_split": args.simulate_historic_split,
                    "output_dir": str(output_dir.relative_to(ROOT_DIR)),
                },
            )
            atomic_write_json(
                output_dir / "run_status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                    "error": None,
                },
            )
            print(f"Dataset integrity audit complete: {completed_at}")
            print(f"Primary result: {output_dir / 'audit.json'}")
        except Exception as exc:
            completed_at = utc_now()
            error_payload = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(
                output_dir / "run_status.json",
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

    atomic_write_json(output_dir / "artifact_manifest.json", build_manifest(output_dir))
    print(f"Artifact manifest: {output_dir / 'artifact_manifest.json'}")


if __name__ == "__main__":
    main()
