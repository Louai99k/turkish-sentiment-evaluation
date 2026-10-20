"""CPU metrics from accepted Teke saved scores; no inference or fitting."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from src.experiment_artifacts import ExperimentRun, sha256_file
from scripts.teke_polarity_inputs import (
    SLOTS, LABELS, PINS, CONTRACT_PIN, csv_rows, input_gate, lineage,
    read_json, require, score_directory,
)

RUN_ID = "professor_round1_teke_polarity_v1"


def calculate(true, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=[0, 1], average=None, zero_division=0)
    return {"n": len(true), "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(np.mean(f1)), "weighted_f1": float(np.average(f1, weights=support)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=[0, 1]).tolist(),
        "per_class": {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])} for i in range(2)}}


def process(run):
    before, contract = input_gate()
    before[Path(__file__).relative_to(ROOT).as_posix()] = sha256_file(Path(__file__))
    run.write_json("input_inventory.json", before)
    try:
        ref = lineage(contract)
        true = np.array([int(r["true_id"]) for r in ref])
        polarity = true < 2
        short = np.array([r["short_text_sensitivity_included"] == "True" for r in ref])
        require(int(polarity.sum()) == 67206 and int((polarity & short).sum()) == 66539, "Polarity supports changed")
        models, table, cells, inputs = [], [], [], []
        fields = ["model", "sample_id", "excel_row", "clean_text_sha256", "true_id", "true_label",
                  "short_text_sensitivity_included", "original_three_class_predicted_id",
                  "score_positive", "score_negative", "score_notr", "constrained_predicted_id",
                  "constrained_predicted_label"]
        with run.path("polarity_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for slot in SLOTS:
                directory = score_directory(slot)
                historical = list(csv_rows(ROOT / "outputs/external" / f"teke_direct_transfer_v1_{slot}/predictions.csv"))
                require(len(historical) == len(ref), "Historical rows missing")
                columns = ("score_positive", "score_negative", "score_notr") if slot == "linear_svm" else (
                    "logit_positive", "logit_negative", "logit_notr")
                predicted = np.full(len(ref), -1, dtype=np.int64)
                count = ties = 0
                for i, row in enumerate(csv_rows(directory / "scores.csv")):
                    require(i < len(ref), "Extra score row")
                    target, old = ref[i], historical[i]
                    for field in ("sample_id", "excel_row", "clean_text_sha256", "true_id", "true_label",
                                  "short_text_sensitivity_included"):
                        require(row[field] == target[field] == old[field], f"Score lineage changed: {field}")
                    values = [float(row[c]) for c in columns]
                    require(all(math.isfinite(v) for v in values) and
                            int(np.argmax(values)) == int(row["predicted_id"]) ==
                            int(row["historical_predicted_id"]) == int(old["predicted_id"]) and
                            row["predicted_label"] == LABELS[int(row["predicted_id"])], "Score reproduction changed")
                    predicted[i] = 0 if values[0] >= values[1] else 1
                    if polarity[i]:
                        ties += int(values[0] == values[1])
                        writer.writerow({"model": slot, **{f: target[f] for f in
                            ("sample_id", "excel_row", "clean_text_sha256", "true_id", "true_label",
                             "short_text_sensitivity_included")},
                            "original_three_class_predicted_id": row["predicted_id"],
                            **dict(zip(("score_positive", "score_negative", "score_notr"), values)),
                            "constrained_predicted_id": int(predicted[i]),
                            "constrained_predicted_label": LABELS[int(predicted[i])]})
                    count += 1
                require(count == 67964 and np.all(predicted >= 0), "Missing score rows")
                views = {}
                for view, mask, expected in (("primary", polarity, 67206),
                                             ("short_text_sensitivity", polarity & short, 66539)):
                    require(int(mask.sum()) == expected, "Wrong evaluation support")
                    metric = calculate(true[mask], predicted[mask])
                    views[view] = metric
                    table.append({"model": slot, "view": view,
                        **{k: metric[k] for k in ("n", "accuracy", "macro_f1", "weighted_f1")},
                        **{f"{label.lower()}_{key}": value for label, record in metric["per_class"].items()
                           for key, value in record.items()}})
                    cells.extend({"model": slot, "view": view, "true_label": LABELS[a],
                        "predicted_label": LABELS[b], "count": metric["confusion_matrix"][a][b]}
                        for a in range(2) for b in range(2))
                models.append({"model": slot, "source_run": read_json(directory / "results.json").get("source_run",
                    "s1_tfidf_svm_winvoker_seed42_base"), "gold_polarity_score_ties": ties, "views": views})
                inputs.append({"model": slot, "path": (directory / "scores.csv").relative_to(ROOT).as_posix(),
                               "sha256": PINS[slot], "score_kind": "decision_margins" if slot == "linear_svm" else "logits"})
                print(f"Calculated {slot}: primary67206 / short66539", flush=True)
        run.write_json("metrics.json", {"schema_version": "teke-constrained-polarity-v1",
            "label_order": LABELS[:2], "tie_rule": "Positive first", "models": models})
        run.write_csv("metric_views.csv", table)
        run.write_csv("confusion_matrices.csv", cells)
        run.write_json("score_inputs.json", inputs)
        run.write_json("results.json", {"status": "metrics_calculated_pending_audit", "models": 4,
            "metric_views": 8, "confusion_cells": 32, "polarity_prediction_rows": 268824,
            "primary_rows_per_model": 67206, "short_rows_per_model": 66539,
            "training": False, "inference": False, "significance_tests": False,
            "scientific_acceptance": "pending_independent_numeric_audit"})
        report = "# Teke constrained polarity — pending audit\n\nGold Positive/Negative only; two-score argmax, "
        report += "Neutral ignored, Positive-first ties. Original frozen checkpoints/masks; no calibration, seed variability, "
        report += "significance or Neutral robustness claim. Margins/logits not calibrated probabilities.\n\n"
        report += "| Model | View | N | Accuracy | Macro F1 | Weighted F1 |\n| --- | --- | ---: | ---: | ---: | ---: |\n"
        for r in table:
            report += f"| {r['model']} | {r['view']} | {r['n']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['weighted_f1']:.6f} |\n"
        run.write_text("report.md", report)
    finally:
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after, "checked_files": len(before)})
        require(before == after, "Protected inputs changed")
    run.declare_complete(["polarity_predictions.csv", "metrics.json", "metric_views.csv", "confusion_matrices.csv",
                          "score_inputs.json", "input_inventory.json", "preservation_check.json", "results.json", "report.md"])


def main():
    output = ROOT / "outputs/analysis" / RUN_ID
    require(not output.exists(), "Refusing existing metric run")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_teke_saved_score_polarity_metrics",
                       config={"score_pins": PINS, "contract_sha256": CONTRACT_PIN,
                               "training": False, "inference": False, "significance_tests": False}) as run:
        process(run)


if __name__ == "__main__":
    main()
