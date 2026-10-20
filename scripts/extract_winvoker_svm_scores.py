"""User-run CPU decision scores from the trusted, frozen historical SVM.

No fitting, model replacement, data download, calibration or performance metrics.
"""
from __future__ import annotations

import csv
import hashlib
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_constants import WINVOKER_TEST_MASK_PATH, WINVOKER_TEST_MASK_SHA256
from scripts.promote_professor_figure_repairs import read_json, require
from scripts.audit_cpu_refit_inventory import parameter

RUN_ID = "professor_round1_winvoker_svm_scores_v1"
INVENTORY = ROOT / "outputs/audits/professor_round1_winvoker_polarity_input_inventory_v1"
HISTORICAL = ROOT / "outputs/experiments/s1_tfidf_svm_winvoker_seed42_base"
BUNDLE_PIN = "0e381fcc40572297b9e792bc2d5cec8eb224dcfdc712feb75455d23a266677d0"
PREDICTION_PIN = "82cbfb870c59e9d69afae7f2c486ee25f542671feba74ff178e99d139926523e"
INVENTORY_PIN = "b1e6bb6c9d0e24681d5bd2b63a90ab07d0a4c9dfc018c6eaf5639250f6678cea"
SCORE_INVENTORY_PIN = "7ac7c41f3ce31790ead4a535a8611890af6f9742253c7559be5ce1be57346d40"
LABELS = ("Positive", "Negative", "Notr")
BATCH_SIZE = 4096


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def inventory_gate():
    require(read_json(INVENTORY / "run_status.json")["status"] == "complete", "Inventory incomplete")
    manifest = read_json(INVENTORY / "artifact_manifest.json")
    require(manifest["run_id"] == INVENTORY.name and manifest["status"] == "complete", "Wrong inventory identity")
    for entry in manifest["files"]:
        path = INVENTORY / entry["name"]
        require(path.resolve().is_relative_to(INVENTORY.resolve()), "Unsafe inventory manifest path")
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"],
                f"Inventory output changed: {path}")
    require(sha256_file(INVENTORY / "input_inventory.json") == INVENTORY_PIN and
            sha256_file(INVENTORY / "score_inventory.json") == SCORE_INVENTORY_PIN, "Accepted inventory pin changed")
    require(read_json(INVENTORY / "results.json")["status"] == "inventory_passed" and
            read_json(INVENTORY / "preservation_check.json")["inputs_unchanged"], "Inventory gates not passed")
    files = read_json(INVENTORY / "input_inventory.json")
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha256_file(path) == expected,
                f"Accepted input changed: {name}")
    # Add accepted gate outputs and this runner to protection; all originals stay read-only.
    for path in [*INVENTORY.iterdir(), Path(__file__).resolve(),
                 ROOT / "scripts/audit_cpu_refit_inventory.py",
                 ROOT / "scripts/promote_professor_figure_repairs.py",
                 ROOT / "src/experiment_artifacts.py"]:
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return files


def state_identity(vectorizer, model):
    vocab = sorted(vectorizer.vocabulary_.items())
    return {
        "vocabulary_sha256": hashlib.sha256(repr(vocab).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
        "coefficient_sha256": hashlib.sha256(model.coef_.tobytes()).hexdigest(),
        "intercept_sha256": hashlib.sha256(model.intercept_.tobytes()).hexdigest(),
    }


def extract(run):
    before = inventory_gate()
    run.write_json("input_inventory.json", before)
    recorded_warnings = []
    try:
        require(sha256_file(HISTORICAL / "model_bundle.pkl") == BUNDLE_PIN and
                sha256_file(HISTORICAL / "predictions.csv") == PREDICTION_PIN and
                sha256_file(WINVOKER_TEST_MASK_PATH) == WINVOKER_TEST_MASK_SHA256, "Frozen SVM/mask identity changed")
        lineage = csv_rows(WINVOKER_TEST_MASK_PATH)
        historic = csv_rows(HISTORICAL / "predictions.csv")
        require(len(lineage) == len(historic) == 88136, "Incomplete input rows")
        for i, (row, ref) in enumerate(zip(historic, lineage)):
            require(int(ref["test_position"]) == i and row["true_label"] == ref["label"] and
                    int(row["true_id"]) == int(ref["label_id"]) and
                    hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == ref["cleaned_text_sha256"],
                    f"Input lineage mismatch at {i}")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            # Only this local artifact, already checked against the accepted SHA, is trusted.
            with (HISTORICAL / "model_bundle.pkl").open("rb") as stream:
                bundle = pickle.load(stream)
            require(set(bundle) >= {"vectorizer", "model"}, "Wrong bundle format")
            vectorizer, model = bundle["vectorizer"], bundle["model"]
            require(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC,
                    "Unexpected frozen estimator types")
            require(np.array_equal(model.classes_, [0, 1, 2]), "Unexpected decision-score class order")
            require(model.multi_class == "ovr" and vectorizer.input == "content", "Unexpected scoring semantics")
            state_before = state_identity(vectorizer, model)
            run.write_json("estimator_record.json", {
                "vectorizer_class": f"{type(vectorizer).__module__}.{type(vectorizer).__name__}",
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "vectorizer_parameters": parameter(vectorizer.get_params(deep=False)),
                "model_parameters": parameter(model.get_params(deep=False)),
                "classes": model.classes_.tolist(), "score_kind": "LinearSVC_ovr_decision_function",
                "column_labels": LABELS, "state_before": state_before,
                "training_version_note": "Current environment recorded; no historical training-manifest repair."})
            disagreements = score_disagreements = polarity_ties = 0
            started = time.perf_counter()
            fields = ["sample_id", "test_position", "source_row_index", "source", "cleaned_text_sha256",
                      "strict_included", "strict_unique_included", "true_id", "true_label",
                      "predicted_id", "predicted_label", "score_positive", "score_negative", "score_notr"]
            with run.path("scores.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for start in range(0, len(historic), BATCH_SIZE):
                    batch = historic[start:start + BATCH_SIZE]
                    matrix = vectorizer.transform([r["text"] for r in batch])
                    scores = np.asarray(model.decision_function(matrix))
                    predicted = np.asarray(model.predict(matrix))
                    require(scores.shape == (len(batch), 3) and np.isfinite(scores).all(), "Invalid decision scores")
                    argmax = model.classes_[np.argmax(scores, axis=1)]
                    for j, (row, score, pred, score_pred) in enumerate(zip(batch, scores, predicted, argmax)):
                        ref = lineage[start + j]
                        disagreements += int(int(pred) != int(row["predicted_id"]))
                        score_disagreements += int(int(score_pred) != int(pred))
                        polarity_ties += int(int(ref["label_id"]) in (0, 1) and score[0] == score[1])
                        writer.writerow({"sample_id": f"winvoker-train-4bd9103ad5f2-{int(ref['source_row_index']):06d}",
                            "test_position": start + j, "source_row_index": ref["source_row_index"],
                            "source": ref["dataset"], "cleaned_text_sha256": ref["cleaned_text_sha256"],
                            "strict_included": ref["strict_included"], "strict_unique_included": ref["strict_unique_included"],
                            "true_id": ref["label_id"], "true_label": ref["label"],
                            "predicted_id": int(pred), "predicted_label": LABELS[int(pred)],
                            "score_positive": float(score[0]), "score_negative": float(score[1]), "score_notr": float(score[2])})
                    print(f"SVM scores: {min(start + BATCH_SIZE, len(historic))}/{len(historic)}", flush=True)
            state_after = state_identity(vectorizer, model)
            run.write_json("reproduction_check.json", {"rows": len(historic),
                "historical_prediction_disagreements": disagreements, "score_argmax_disagreements": score_disagreements,
                "gold_polarity_score_ties": polarity_ties, "state_unchanged": state_before == state_after,
                "state_after": state_after, "score_seconds": time.perf_counter() - started})
            require(disagreements == score_disagreements == 0, "Historical predictions not reproduced; investigate, do not replace")
            require(state_before == state_after, "Frozen estimator state changed")
            recorded_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        run.write_json("results.json", {"status": "scores_extracted", "rows": 88136,
            "source_run": HISTORICAL.name, "source_bundle_sha256": BUNDLE_PIN,
            "source_predictions_sha256": PREDICTION_PIN, "mask_sha256": WINVOKER_TEST_MASK_SHA256,
            "score_kind": "LinearSVC_ovr_decision_function", "score_order": LABELS,
            "tie_rule": "Positive first for constrained Positive/Negative argmax",
            "training": False, "calibration": False, "performance_metrics": False,
            "external_artifacts_modified": False, "scientific_acceptance": "pending_review"})
        run.write_text("report.md", "# Frozen historical SVM score extraction\n\nCPU transform/decision_function/predict only; three-class scores saved with stable mask-derived lineage, no raw texts copied. Historical hard labels and score argmax reproduced; learned state unchanged. Scores are decision margins, not probabilities or neural logits. No fit/calibration/checkpoint substitution. No retroactive historical provenance repair or external result replacement. Constrained metrics remain a separate next gate.\n")
    finally:
        # Preserve warnings even when extraction fails inside the capture context.
        if 'captured' in locals():
            recorded_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        run.write_json("warnings.json", recorded_warnings)
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after,
                                                   "checked_files": len(before)})
        require(before == after, "Protected input changed")
    run.declare_complete(["scores.csv", "input_inventory.json", "estimator_record.json",
                          "reproduction_check.json", "warnings.json", "preservation_check.json",
                          "results.json", "report.md"])


def main():
    output = ROOT / "outputs/analysis" / RUN_ID
    require(not output.exists(), "Refusing existing score extraction run")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_frozen_svm_decision_score_extraction",
                       config={"source_run": HISTORICAL.name, "bundle_pin": BUNDLE_PIN,
                               "prediction_pin": PREDICTION_PIN, "input_inventory_pin": INVENTORY_PIN,
                               "mask_pin": WINVOKER_TEST_MASK_SHA256, "batch_size": BATCH_SIZE,
                               "training": False, "device": "cpu", "performance_metrics": False}) as run:
        extract(run)


if __name__ == "__main__":
    main()
