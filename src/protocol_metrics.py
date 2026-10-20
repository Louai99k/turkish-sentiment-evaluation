"""Canonical protocol-v2 classification metrics and diagnostic views."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.protocol_constants import ID_TO_LABEL, LABEL_IDS, METRICS_SCHEMA_VERSION
from src.protocol_predictions import normalize_label_ids, parse_boolean_series


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    subset_id: str,
    averaging_label_ids: Iterable[int] = LABEL_IDS,
    interpretation: str = "classification",
) -> dict[str, Any]:
    """Compute the one canonical unrounded metric payload.

    Per-class values and the confusion matrix always use the three canonical
    labels. Macro/weighted averages use ``averaging_label_ids`` so polarity and
    single-label source diagnostics are not diluted with unsupported classes.
    """

    true = normalize_label_ids(list(y_true), f"{subset_id}.true")
    pred = normalize_label_ids(list(y_pred), f"{subset_id}.predicted")
    if len(true) != len(pred):
        raise ValueError(f"True/predicted lengths differ for {subset_id}")
    if len(true) == 0:
        raise ValueError(f"Cannot calculate metrics for empty subset {subset_id}")

    average_ids = tuple(int(value) for value in averaging_label_ids)
    if not average_ids or len(set(average_ids)) != len(average_ids):
        raise ValueError(f"Invalid averaging labels for {subset_id}: {average_ids}")
    invalid = sorted(set(average_ids) - set(LABEL_IDS))
    if invalid:
        raise ValueError(f"Unknown averaging label IDs for {subset_id}: {invalid}")

    macro = precision_recall_fscore_support(
        true,
        pred,
        labels=list(average_ids),
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        true,
        pred,
        labels=list(average_ids),
        average="weighted",
        zero_division=0,
    )
    per_class_values = precision_recall_fscore_support(
        true,
        pred,
        labels=list(LABEL_IDS),
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for position, label_id in enumerate(LABEL_IDS):
        per_class[ID_TO_LABEL[label_id]] = {
            "precision": float(per_class_values[0][position]),
            "recall": float(per_class_values[1][position]),
            "f1": float(per_class_values[2][position]),
            "support": int(per_class_values[3][position]),
        }

    prediction_rates = {
        ID_TO_LABEL[label_id]: float(np.mean(pred == label_id)) for label_id in LABEL_IDS
    }
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "subset_id": subset_id,
        "interpretation": interpretation,
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "negative_f1": float(per_class["Negative"]["f1"]),
        "notr_prediction_rate": prediction_rates["Notr"],
        "off_averaging_label_prediction_rate": float(
            np.mean(~np.isin(pred, np.asarray(average_ids, dtype=np.int64)))
        ),
        "averaging_label_ids": list(average_ids),
        "averaging_labels": [ID_TO_LABEL[value] for value in average_ids],
        "prediction_rates": prediction_rates,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(true, pred, labels=list(LABEL_IDS)).tolist(),
    }


def _source_averaging_labels(true_ids: np.ndarray) -> tuple[int, ...]:
    supported = set(int(value) for value in np.unique(true_ids))
    if supported <= {0, 1}:
        return (0, 1)
    if supported == {2}:
        return (2,)
    return tuple(label_id for label_id in LABEL_IDS if label_id in supported)


def build_evaluation_metrics(
    predictions: pd.DataFrame,
    *,
    primary_view: str = "historic_test_complete",
) -> dict[str, Any]:
    required = {"true_id", "predicted_id", "strict_included", "source"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing metric-view columns: {sorted(missing)}")

    frame = predictions.reset_index(drop=True).copy()
    frame["true_id"] = normalize_label_ids(frame["true_id"], "predictions.true_id")
    frame["predicted_id"] = normalize_label_ids(
        frame["predicted_id"], "predictions.predicted_id"
    )
    frame["strict_included"] = parse_boolean_series(
        frame["strict_included"], "predictions.strict_included"
    )
    frame["source"] = frame["source"].astype(str)

    views: dict[str, dict[str, Any]] = {}

    def add_view(
        view_id: str,
        mask: np.ndarray | pd.Series,
        averaging_ids: Iterable[int],
        interpretation: str,
        source: str | None = None,
    ) -> None:
        selected = frame.loc[np.asarray(mask, dtype=bool)]
        if selected.empty:
            return
        metrics = classification_metrics(
            selected["true_id"],
            selected["predicted_id"],
            subset_id=view_id,
            averaging_label_ids=averaging_ids,
            interpretation=interpretation,
        )
        metrics["source"] = source
        views[view_id] = metrics

    all_rows = np.ones(len(frame), dtype=bool)
    strict = frame["strict_included"].to_numpy(dtype=bool)
    polarity = frame["true_id"].isin([0, 1]).to_numpy(dtype=bool)
    add_view(
        "historic_test_complete",
        all_rows,
        LABEL_IDS,
        "continuity_headline",
    )
    add_view(
        "historic_test_strict",
        strict,
        LABEL_IDS,
        "prespecified_exact_leakage_sensitivity",
    )
    add_view(
        "polarity_only_complete",
        polarity,
        (0, 1),
        "source_confound_diagnostic",
    )
    add_view(
        "polarity_only_strict",
        polarity & strict,
        (0, 1),
        "source_confound_diagnostic",
    )

    for source in sorted(frame["source"].unique()):
        source_mask = frame["source"].eq(source).to_numpy(dtype=bool)
        source_labels = _source_averaging_labels(
            frame.loc[source_mask, "true_id"].to_numpy(dtype=np.int64)
        )
        add_view(
            f"source_complete:{source}",
            source_mask,
            source_labels,
            "source_specific_diagnostic",
            source,
        )
        add_view(
            f"source_strict:{source}",
            source_mask & strict,
            source_labels,
            "source_specific_diagnostic",
            source,
        )

    if primary_view not in views:
        raise ValueError(f"Primary metric view was not produced: {primary_view}")
    primary = views[primary_view]
    document = {
        key: value
        for key, value in primary.items()
        if key not in {"schema_version", "subset_id", "interpretation", "source"}
    }
    document.update(
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "primary_view": primary_view,
            "label_order": [ID_TO_LABEL[value] for value in LABEL_IDS],
            "views": views,
        }
    )
    return document


def flatten_metric_views(metrics_document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_id, metrics in metrics_document["views"].items():
        row = {
            "subset_id": view_id,
            "interpretation": metrics["interpretation"],
            "source": metrics.get("source"),
            "n": metrics["n"],
            "averaging_labels": "|".join(metrics["averaging_labels"]),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "negative_f1": metrics["negative_f1"],
            "notr_prediction_rate": metrics["notr_prediction_rate"],
            "off_averaging_label_prediction_rate": metrics[
                "off_averaging_label_prediction_rate"
            ],
        }
        for label_name in ("Positive", "Negative", "Notr"):
            prefix = label_name.lower()
            values = metrics["per_class"][label_name]
            row[f"{prefix}_precision"] = values["precision"]
            row[f"{prefix}_recall"] = values["recall"]
            row[f"{prefix}_f1"] = values["f1"]
            row[f"{prefix}_support"] = values["support"]
        rows.append(row)
    return rows


def confusion_matrix_rows(metrics_document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_id, metrics in metrics_document["views"].items():
        matrix = metrics["confusion_matrix"]
        for true_position, true_id in enumerate(LABEL_IDS):
            for predicted_position, predicted_id in enumerate(LABEL_IDS):
                rows.append(
                    {
                        "subset_id": view_id,
                        "true_id": true_id,
                        "true_label": ID_TO_LABEL[true_id],
                        "predicted_id": predicted_id,
                        "predicted_label": ID_TO_LABEL[predicted_id],
                        "count": int(matrix[true_position][predicted_position]),
                    }
                )
    return rows
