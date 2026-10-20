"""Build and validate protocol-v2 prediction records with stable lineage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.protocol_constants import (
    ID_TO_LABEL,
    LABEL_IDS,
    LABEL_TO_ID,
    PREDICTIONS_SCHEMA_VERSION,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
    WINVOKER_STRICT_TEST_ROWS,
    WINVOKER_TEST_MASK_PATH,
    WINVOKER_TEST_MASK_SHA256,
    WINVOKER_TEST_ROWS,
    WINVOKER_TEST_SPLIT_ID,
    WINVOKER_UPSTREAM_SPLIT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def parse_boolean_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        invalid = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Invalid Boolean values in {name}: {invalid}")
    return mapped.astype(bool)


def normalize_label_ids(values: Sequence[Any], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")

    if array.dtype.kind in {"U", "S", "O"}:
        text_values = pd.Series(array).astype(str).str.strip()
        converted = text_values.map(LABEL_TO_ID)
        numeric_fallback = pd.to_numeric(text_values, errors="coerce")
        converted = converted.where(converted.notna(), numeric_fallback)
    else:
        converted = pd.to_numeric(pd.Series(array), errors="coerce")

    if converted.isna().any() or not np.equal(converted, np.floor(converted)).all():
        invalid = sorted(pd.Series(array)[converted.isna()].astype(str).unique())
        detail = f": {invalid}" if invalid else ""
        raise ValueError(f"{name} contains unknown, missing, or non-integral labels{detail}")
    array = converted.to_numpy(dtype=np.int64)

    invalid_ids = sorted(set(array.tolist()) - set(LABEL_IDS))
    if invalid_ids:
        raise ValueError(f"Unknown label IDs in {name}: {invalid_ids}")
    return array


def load_winvoker_test_lineage(
    path: Path | str = WINVOKER_TEST_MASK_PATH,
    verify_hash: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical Winvoker test mask not found: {path}")
    if verify_hash:
        actual_hash = sha256_file(path)
        if actual_hash != WINVOKER_TEST_MASK_SHA256:
            raise ValueError(
                "Canonical Winvoker test-mask hash mismatch: "
                f"expected {WINVOKER_TEST_MASK_SHA256}, got {actual_hash}"
            )

    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "test_position",
        "source_row_index",
        "label",
        "label_id",
        "dataset",
        "strict_included",
        "strict_unique_included",
        "cleaned_text_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test mask is missing columns: {sorted(missing)}")
    if len(frame) != WINVOKER_TEST_ROWS:
        raise ValueError(
            f"Expected {WINVOKER_TEST_ROWS} test rows, found {len(frame)} in {path}"
        )

    positions = pd.to_numeric(frame["test_position"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(WINVOKER_TEST_ROWS, dtype=np.int64)):
        raise ValueError("Canonical test positions are not contiguous and ordered")

    frame = frame.copy()
    frame["label_id"] = normalize_label_ids(frame["label_id"], "lineage.label_id")
    expected_names = pd.Series(frame["label_id"]).map(ID_TO_LABEL).to_numpy()
    if not np.array_equal(expected_names, frame["label"].astype(str).to_numpy()):
        raise ValueError("Canonical test-mask label IDs and names disagree")
    frame["strict_included"] = parse_boolean_series(
        frame["strict_included"], "lineage.strict_included"
    )
    frame["strict_unique_included"] = parse_boolean_series(
        frame["strict_unique_included"], "lineage.strict_unique_included"
    )
    if int(frame["strict_included"].sum()) != WINVOKER_STRICT_TEST_ROWS:
        raise ValueError("Canonical strict-test support does not match the locked protocol")
    return frame


def stable_winvoker_sample_id(source_row_index: int) -> str:
    revision = WINVOKER_DATASET_REVISION[:12]
    return f"winvoker-train-{revision}-{int(source_row_index):06d}"


def build_winvoker_prediction_frame(
    lineage: pd.DataFrame,
    cleaned_texts: Sequence[Any],
    true_labels: Sequence[Any],
    predicted_labels: Sequence[Any],
    scores: np.ndarray | None = None,
    score_kind: str | None = None,
) -> pd.DataFrame:
    """Create the canonical predictions table in historical test order.

    ``scores`` must use canonical class order ``[Positive, Negative, Notr]``.
    ``score_kind`` should describe the values accurately, for example ``logit``,
    ``probability``, or ``decision_score``.
    """

    texts = pd.Series(cleaned_texts, dtype="object").fillna("").astype(str).reset_index(drop=True)
    y_true = normalize_label_ids(true_labels, "true_labels")
    y_pred = normalize_label_ids(predicted_labels, "predicted_labels")
    if not (len(lineage) == len(texts) == len(y_true) == len(y_pred)):
        raise ValueError(
            "Prediction inputs differ in length: "
            f"lineage={len(lineage)}, texts={len(texts)}, true={len(y_true)}, pred={len(y_pred)}"
        )

    expected_true = lineage["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_true, expected_true):
        mismatch = int(np.flatnonzero(y_true != expected_true)[0])
        raise ValueError(f"True-label lineage mismatch at test position {mismatch}")

    text_hashes = texts.map(sha256_text)
    expected_hashes = lineage["cleaned_text_sha256"].astype(str).reset_index(drop=True)
    if not text_hashes.equals(expected_hashes):
        mismatch = int(np.flatnonzero(text_hashes.to_numpy() != expected_hashes.to_numpy())[0])
        raise ValueError(f"Cleaned-text lineage mismatch at test position {mismatch}")

    source_rows = pd.to_numeric(lineage["source_row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    frame = pd.DataFrame(
        {
            "prediction_schema_version": PREDICTIONS_SCHEMA_VERSION,
            "sample_id": [stable_winvoker_sample_id(value) for value in source_rows],
            "dataset_id": WINVOKER_DATASET_ID,
            "dataset_revision": WINVOKER_DATASET_REVISION,
            "upstream_split": WINVOKER_UPSTREAM_SPLIT,
            "benchmark_split_id": WINVOKER_TEST_SPLIT_ID,
            "test_position": lineage["test_position"].to_numpy(dtype=np.int64),
            "source_row_index": source_rows,
            "source": lineage["dataset"].astype(str).to_numpy(),
            "strict_included": lineage["strict_included"].to_numpy(dtype=bool),
            "strict_unique_included": lineage["strict_unique_included"].to_numpy(dtype=bool),
            "cleaned_text_sha256": text_hashes.to_numpy(),
            "text": texts.to_numpy(),
            "true_id": y_true,
            "predicted_id": y_pred,
            "true_label": [ID_TO_LABEL[value] for value in y_true],
            "predicted_label": [ID_TO_LABEL[value] for value in y_pred],
            "prediction_correct": y_true == y_pred,
        }
    )

    if scores is not None:
        values = np.asarray(scores)
        if values.shape != (len(frame), len(LABEL_IDS)):
            raise ValueError(
                "Scores must have shape "
                f"({len(frame)}, {len(LABEL_IDS)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Scores contain NaN or infinite values")
        if not score_kind or not str(score_kind).strip():
            raise ValueError("score_kind is required when scores are provided")
        prefix = str(score_kind).strip().lower().replace(" ", "_")
        for position, label_id in enumerate(LABEL_IDS):
            frame[f"{prefix}_{ID_TO_LABEL[label_id].lower()}"] = values[:, position]

    if not frame["sample_id"].is_unique:
        raise ValueError("Stable sample IDs are not unique")
    return frame
