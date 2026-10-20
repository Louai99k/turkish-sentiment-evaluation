"""User-run constrained two-score metrics from frozen saved scores only.

No model loading, inference, fitting, calibration, mask changes or significance tests.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_constants import WINVOKER_TEST_MASK_PATH, WINVOKER_TEST_MASK_SHA256
from scripts.promote_professor_figure_repairs import require, read_json, verify_manifest
from scripts.extract_winvoker_svm_scores import inventory_gate
from scripts.audit_winvoker_polarity_inputs import ROSTER, PREDICTION_PINS

RUN_ID = "professor_round1_winvoker_polarity_v1"
SVM = ROOT / "outputs/analysis/professor_round1_winvoker_svm_scores_v1"
SVM_SCORE_PIN = "bcbac9d0c7200121c700853b9e315937a30e21bfdf4c1952ffc7b8bfce7f35cf"
INVENTORY = ROOT / "outputs/audits/professor_round1_winvoker_polarity_input_inventory_v1"
LABELS = ("Positive", "Negative")
SOURCES = ("HUMIR", "urun_yorumlari", "magaza_yorumlari", "tweet-pn")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def calculate(true, predicted, identifier):
    require(len(true) > 0 and set(true) <= {0, 1} and set(predicted) <= {0, 1}, "Invalid polarity view")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=[0, 1], average=None, zero_division=0)
    cm = confusion_matrix(true, predicted, labels=[0, 1])
    require(int(cm.sum()) == len(true), "Confusion support mismatch")
    return {"view_id": identifier, "n": len(true), "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(np.mean(f1)), "weighted_f1": float(np.average(f1, weights=support)),
        "label_order": LABELS, "zero_division": 0, "confusion_matrix": cm.tolist(),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(LABELS)}}


def process(run):
    before = inventory_gate()
    verify_manifest(SVM)
    require(sha256_file(SVM / "scores.csv") == SVM_SCORE_PIN, "Accepted SVM scores changed")
    reproduction = read_json(SVM / "reproduction_check.json")
    require(reproduction["historical_prediction_disagreements"] == 0 and
            reproduction["score_argmax_disagreements"] == 0 and reproduction["state_unchanged"] and
            read_json(SVM / "preservation_check.json")["inputs_unchanged"], "SVM extraction gates failed")
    # Recheck every protected score-extraction input, not just its output manifest.
    for name, expected in read_json(SVM / "input_inventory.json").items():
        require(sha256_file(ROOT / name) == expected, f"Changed score-extraction input: {name}")
        before[name] = expected
    for path in [*SVM.iterdir(), Path(__file__).resolve(),
                 ROOT / "scripts/audit_winvoker_polarity_inputs.py"]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    run.write_json("input_inventory.json", before)
    try:
        require(sha256_file(WINVOKER_TEST_MASK_PATH) == WINVOKER_TEST_MASK_SHA256, "Mask changed")
        lineage = list(csv_rows(WINVOKER_TEST_MASK_PATH))
        require(len(lineage) == 88136, "Wrong full lineage support")
        true = np.array([int(r["label_id"]) for r in lineage])
        polarity = np.isin(true, [0, 1])
        require(all(r["strict_included"] in ("True", "False") for r in lineage), "Invalid mask boolean")
        strict = np.array([r["strict_included"] == "True" for r in lineage])
        sources = np.array([r["dataset"] for r in lineage])
        require(int(polarity.sum()) == 57371 and int((polarity & strict).sum()) == 56870, "Polarity support changed")
        require(set(sources[polarity]) == set(SOURCES), "Unexpected polarity sources")
        support_reference = read_json(INVENTORY / "lineage_summary.json")
        all_metrics, metric_table, confusions, score_inputs = [], [], [], []
        fields = ["model", "source_run", "score_kind", "sample_id", "test_position", "source",
                  "cleaned_text_sha256", "strict_included", "true_id", "true_label",
                  "original_three_class_predicted_id", "score_positive", "score_negative", "score_notr",
                  "constrained_predicted_id", "constrained_predicted_label"]
        with run.path("polarity_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in ROSTER:
                slot = item["slot"]
                svm = slot == "linear_svm"
                path = SVM / "scores.csv" if svm else ROOT / "outputs/experiments" / item["run"] / "predictions.csv"
                pin = SVM_SCORE_PIN if svm else PREDICTION_PINS[slot]
                require(sha256_file(path) == pin, "Score input changed")
                columns = ("score_positive", "score_negative", "score_notr") if svm else (
                    "logit_positive", "logit_negative", "logit_notr")
                kind = "LinearSVC_ovr_decision_function" if svm else "neural_logits"
                predicted = np.full(len(lineage), -1, dtype=np.int64)
                ties = 0
                count = 0
                for position, row in enumerate(csv_rows(path)):
                    require(position < len(lineage), "Extra score row")
                    ref = lineage[position]
                    sample_id = f"winvoker-train-4bd9103ad5f2-{int(ref['source_row_index']):06d}"
                    require(row["sample_id"] == sample_id and int(row["test_position"]) == position and
                            int(row["true_id"]) == int(ref["label_id"]) and row["true_label"] == ref["label"] and
                            row["source"] == ref["dataset"] and row["source_row_index"] == ref["source_row_index"] and
                            row["cleaned_text_sha256"] == ref["cleaned_text_sha256"] and
                            row["strict_included"] == ref["strict_included"] and
                            row["strict_unique_included"] == ref["strict_unique_included"], "Score lineage mismatch")
                    if not svm:
                        require(hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == ref["cleaned_text_sha256"],
                                "Neural text lineage mismatch")
                    values = np.array([float(row[c]) for c in columns])
                    require(np.isfinite(values).all() and int(np.argmax(values)) == int(row["predicted_id"]),
                            "Score finiteness/three-class reproduction mismatch")
                    predicted[position] = 0 if values[0] >= values[1] else 1
                    if polarity[position]:
                        ties += int(values[0] == values[1])
                        writer.writerow({"model": slot, "source_run": item["run"], "score_kind": kind,
                            "sample_id": sample_id, "test_position": position, "source": ref["dataset"],
                            "cleaned_text_sha256": ref["cleaned_text_sha256"], "strict_included": ref["strict_included"],
                            "true_id": int(ref["label_id"]), "true_label": ref["label"],
                            "original_three_class_predicted_id": int(row["predicted_id"]),
                            "score_positive": float(values[0]), "score_negative": float(values[1]), "score_notr": float(values[2]),
                            "constrained_predicted_id": int(predicted[position]),
                            "constrained_predicted_label": LABELS[predicted[position]]})
                    count += 1
                require(count == 88136 and np.all(predicted >= 0), "Missing score rows")
                model_views = {}
                for view, mask in (("complete", polarity), ("strict", polarity & strict)):
                    model_views[view] = {}
                    for source in ("all", *SOURCES):
                        selected = mask if source == "all" else mask & (sources == source)
                        expected = (57371 if view == "complete" else 56870) if source == "all" else support_reference["polarity_by_source"][view][source]
                        require(int(selected.sum()) == expected, "Source support mismatch")
                        metric = calculate(true[selected], predicted[selected], f"{view}:{source}")
                        model_views[view][source] = metric
                        metric_table.append({"model": slot, "view": view, "source": source, "n": metric["n"],
                            **{key: metric[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
                            **{f"{label.lower()}_{key}": value for label, record in metric["per_class"].items()
                               for key, value in record.items()}})
                        confusions.extend({"model": slot, "view": view, "source": source, "true_label": LABELS[i],
                                           "predicted_label": LABELS[j], "count": metric["confusion_matrix"][i][j]}
                                          for i in range(2) for j in range(2))
                    total = np.sum([model_views[view][s]["confusion_matrix"] for s in SOURCES], axis=0)
                    require(np.array_equal(total, model_views[view]["all"]["confusion_matrix"]), "Source confusions do not reconcile")
                all_metrics.append({"model": slot, "source_run": item["run"], "score_kind": kind,
                                    "gold_polarity_score_ties": ties, "views": model_views})
                score_inputs.append({"model": slot, "source_run": item["run"], "path": path.relative_to(ROOT).as_posix(),
                                     "sha256": pin, "score_kind": kind})
                print(f"Calculated {slot}: complete 57371 / strict 56870; four source views", flush=True)
        run.write_json("metrics.json", {"schema_version": "constrained-polarity-metrics-v1",
            "label_order": LABELS, "tie_rule": "Positive first", "models": all_metrics})
        run.write_csv("metric_views.csv", metric_table)
        run.write_csv("confusion_matrices.csv", confusions)
        run.write_json("score_inputs.json", score_inputs)
        run.write_json("results.json", {"status": "metrics_calculated_pending_audit", "models": 4,
            "metric_views": len(metric_table), "polarity_prediction_rows": 4 * 57371,
            "complete_rows_per_model": 57371, "strict_rows_per_model": 56870,
            "frozen_mask_sha256": WINVOKER_TEST_MASK_SHA256, "training": False, "inference": False,
            "significance_tests": False, "scientific_acceptance": "pending_independent_numeric_audit"})
        report = "# Constrained Winvoker polarity metrics — pending audit\n\nGold Positive/Negative only; argmax over their two scores, Neutral ignored. Positive wins exact ties. Frozen external roster, complete and original strict mask; no calibration/selection/model execution. Neural logits and SVM decision margins are not calibrated across models. Two-class Macro/Weighted F1 use ONLY Positive/Negative; confusion rows=true, columns=predicted. Source confusions reconcile to aggregate; wiki/random omitted for absent gold polarity support. No significance or seed variance inferred; source support/imbalance and dataset source cues limit generalization. Research-only outputs, not public repository assets.\n\n| Model | View | N | Accuracy | Macro F1 | Weighted F1 |\n| --- | --- | ---: | ---: | ---: | ---: |\n"
        for row in metric_table:
            if row["source"] == "all":
                report += f"| {row['model']} | {row['view']} | {row['n']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['weighted_f1']:.6f} |\n"
        run.write_text("report.md", report)
    finally:
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after, "checked_files": len(before)})
        require(before == after, "Protected input changed")
    run.declare_complete(["polarity_predictions.csv", "metrics.json", "metric_views.csv", "confusion_matrices.csv",
                          "score_inputs.json", "input_inventory.json", "preservation_check.json", "results.json", "report.md"])


def main():
    output = ROOT / "outputs/analysis" / RUN_ID
    require(not output.exists(), "Refusing existing polarity run")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_saved_score_constrained_polarity_metrics",
                       config={"roster": ROSTER, "neural_prediction_pins": PREDICTION_PINS,
                               "svm_score_pin": SVM_SCORE_PIN, "mask_pin": WINVOKER_TEST_MASK_SHA256,
                               "label_order": LABELS, "sources": SOURCES, "tie_rule": "Positive first",
                               "training": False, "inference": False, "significance_tests": False}) as run:
        process(run)


if __name__ == "__main__":
    main()
