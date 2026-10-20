"""R09 trusted original SVM CPU score extraction; no fit/calibration/metrics."""
from __future__ import annotations

import csv
import hashlib
import json
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

from scripts.audit_cpu_refit_inventory import parameter
from scripts.run_teke_direct_transfer import load_masked_rows
from src.experiment_artifacts import ExperimentRun, sha256_file

RUN_ID = "professor_round1_teke_scores_v1_linear_svm"
PREFLIGHT = ROOT / "outputs/audits/professor_round1_teke_score_preflight_v1"
CONTRACT = ROOT / "configs/analysis/professor_round1_teke_score_contract_v1.json"
REPORT_PIN = "1f6d471f0d81655d93283ec7e40ef19f4a50932677012b0ea81a091de96c0dee"
INVENTORY_PIN = "7daccf388787d56c3896d748e1e98cbea5354ddce9415ea85fb6c009f76d191a"
CONTRACT_PIN = "e1346f81cdd152a72a4e496761616eb4dcd816079a25958708305b1b84a11af5"
HISTORICAL = ROOT / "outputs/external/teke_direct_transfer_v1_linear_svm"
BUNDLE = ROOT / "outputs/experiments/s1_tfidf_svm_winvoker_seed42_base/model_bundle.pkl"
LABELS = ("Positive", "Negative", "Notr")
BATCH_SIZE = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate():
    status = read_json(PREFLIGHT / "run_status.json")
    require(status["run_id"] == PREFLIGHT.name and status["status"] == "complete" and
            status["error"] is None, "Accepted preflight incomplete")
    manifest = read_json(PREFLIGHT / "artifact_manifest.json")
    require(manifest["run_id"] == PREFLIGHT.name and manifest["status"] == "complete",
            "Preflight manifest identity mismatch")
    names = set()
    for entry in manifest["files"]:
        name = entry["name"]
        require(Path(name).name == name and name not in names, "Unsafe/duplicate manifest name")
        names.add(name)
        path = PREFLIGHT / name
        require(path.stat().st_size == entry["size_bytes"] and
                sha256_file(path) == entry["sha256"], f"Preflight payload changed: {name}")
    require(sha256_file(PREFLIGHT / "report.json") == REPORT_PIN and
            sha256_file(PREFLIGHT / "input_inventory.json") == INVENTORY_PIN and
            sha256_file(CONTRACT) == CONTRACT_PIN, "Accepted gate/contract pin changed")
    report = read_json(PREFLIGHT / "report.json")
    preservation = read_json(PREFLIGHT / "preservation_check.json")
    before = read_json(PREFLIGHT / "input_inventory.json")
    require(preservation["all_inputs_unchanged"] is True and
            preservation["before"] == preservation["after"] == before and
            report["contract_sha256"] == CONTRACT_PIN and
            report["scientific_status"] == "static_artifacts_and_lineage_checked_no_model_loaded",
            "Accepted preflight scientific/preservation gate mismatch")
    for name, expected in before.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha256_file(path) == expected,
                f"Frozen preflight input changed: {name}")
    # Extend protection without editing any accepted manifest.
    for path in [*PREFLIGHT.iterdir(), Path(__file__).resolve(),
                 ROOT / "scripts/audit_cpu_refit_inventory.py",
                 ROOT / "src/experiment_artifacts.py", ROOT / "src/protocol_constants.py"]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return before, read_json(CONTRACT)


def state_identity(vectorizer, model):
    return {
        "vocabulary_sha256": hashlib.sha256(
            repr(sorted(vectorizer.vocabulary_.items())).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
        "coefficient_sha256": hashlib.sha256(model.coef_.tobytes()).hexdigest(),
        "intercept_sha256": hashlib.sha256(model.intercept_.tobytes()).hexdigest(),
    }


def extract(run):
    before, contract = gate()
    run.write_json("input_inventory.json", before)
    captured = []
    try:
        pin = contract["historical"]["linear_svm"]
        require(sha256_file(BUNDLE) == pin["checkpoint_sha256"], "Original SVM bundle changed")
        require(sha256_file(HISTORICAL / "predictions.csv") == pin["payloads"]["predictions.csv"],
                "Historical Teke predictions changed")
        # Existing loader decodes pinned ZIP/XLSX, joins frozen mask and checks
        # canonical profile/legacy clean_text equivalence on every primary row.
        rows, support = load_masked_rows()
        require(support["primary"] == contract["primary_support"] and
                support["short_text_sensitivity"] == contract["short_support"], "Support mismatch")
        with (HISTORICAL / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            historic = list(csv.DictReader(stream))
        require(len(rows) == len(historic) == 67964, "Primary row count mismatch")
        ordered_sha = hashlib.sha256("\n".join(r["sample_id"] for r in rows).encode("utf-8")).hexdigest()
        require(ordered_sha == contract["ordered_sample_ids_sha256"], "Loaded row ordering changed")
        for row, old in zip(rows, historic):
            require(row["sample_id"] == old["sample_id"] and
                    int(row["excel_row"]) == int(old["excel_row"]) and
                    row["clean_text_sha256"] == old["clean_text_sha256"] and
                    row["true_label"] == old["true_label"] and
                    LABELS.index(row["true_label"]) == int(old["true_id"]) and
                    row["short_text_sensitivity_included"] ==
                    (old["short_text_sensitivity_included"] == "True"), "Loaded row lineage mismatch")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            # Trusted local pickle only, after exact accepted artifact hash check.
            with BUNDLE.open("rb") as stream:
                bundle = pickle.load(stream)
            require(set(bundle) >= {"model", "vectorizer"}, "Unexpected bundle schema")
            vectorizer, model = bundle["vectorizer"], bundle["model"]
            require(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC,
                    "Unexpected estimator types")
            require(np.array_equal(model.classes_, [0, 1, 2]) and
                    model.multi_class == "ovr" and vectorizer.input == "content",
                    "Unexpected class order/scoring semantics")
            state_before = state_identity(vectorizer, model)
            run.write_json("estimator_record.json", {
                "source_run": "s1_tfidf_svm_winvoker_seed42_base",
                "bundle_sha256": pin["checkpoint_sha256"],
                "classes": model.classes_.tolist(), "score_order": LABELS,
                "vectorizer_parameters": parameter(vectorizer.get_params(deep=False)),
                "model_parameters": parameter(model.get_params(deep=False)),
                "state_before": state_before, "score_kind": "LinearSVC_ovr_decision_function",
                "historical_training_provenance_repaired": False})
            disagreements = score_disagreements = ties = 0
            score_started = time.perf_counter()
            fields = ["sample_id", "excel_row", "clean_text_sha256", "true_label", "true_id",
                      "short_text_sensitivity_included", "historical_predicted_id",
                      "predicted_id", "predicted_label", "score_positive", "score_negative", "score_notr"]
            with run.path("scores.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for start in range(0, len(rows), BATCH_SIZE):
                    batch = rows[start:start + BATCH_SIZE]
                    matrix = vectorizer.transform([r["text"] for r in batch])
                    scores = np.asarray(model.decision_function(matrix))
                    predictions = np.asarray(model.predict(matrix))
                    require(scores.shape == (len(batch), 3) and np.isfinite(scores).all(),
                            "Invalid/nonfinite decision scores")
                    argmax = model.classes_[np.argmax(scores, axis=1)]
                    for j, (row, score, pred, score_pred) in enumerate(zip(batch, scores, predictions, argmax)):
                        old = historic[start + j]
                        disagreements += int(int(pred) != int(old["predicted_id"]))
                        score_disagreements += int(int(pred) != int(score_pred))
                        ties += int(row["true_label"] in LABELS[:2] and score[0] == score[1])
                        writer.writerow({"sample_id": row["sample_id"], "excel_row": row["excel_row"],
                            "clean_text_sha256": row["clean_text_sha256"], "true_label": row["true_label"],
                            "true_id": LABELS.index(row["true_label"]),
                            "short_text_sensitivity_included": row["short_text_sensitivity_included"],
                            "historical_predicted_id": old["predicted_id"], "predicted_id": int(pred),
                            "predicted_label": LABELS[int(pred)], "score_positive": float(score[0]),
                            "score_negative": float(score[1]), "score_notr": float(score[2])})
                    print(f"Teke original SVM scores: {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
            state_after = state_identity(vectorizer, model)
            run.write_json("reproduction_check.json", {"rows": len(rows),
                "historical_prediction_disagreements": disagreements,
                "score_argmax_disagreements": score_disagreements,
                "gold_polarity_score_ties": ties, "state_unchanged": state_before == state_after,
                "state_after": state_after, "score_seconds": time.perf_counter() - score_started})
            require(disagreements == score_disagreements == 0,
                    "Historical labels not reproduced; investigate, do not replace")
            require(state_before == state_after, "Frozen learned estimator state changed")
        run.write_json("results.json", {"status": "scores_extracted_pending_review",
            "rows": len(rows), "primary_support": support["primary"],
            "short_support": support["short_text_sensitivity"],
            "ordered_sample_ids_sha256": ordered_sha, "contract_sha256": CONTRACT_PIN,
            "source_zip_sha256": contract["source_zip_sha256"], "mask_sha256": contract["mask_sha256"],
            "profile_sha256": contract["profile_sha256"], "bundle_sha256": pin["checkpoint_sha256"],
            "historical_predictions_sha256": pin["payloads"]["predictions.csv"],
            "score_order": LABELS, "score_kind": "LinearSVC_ovr_decision_function",
            "training": False, "calibration": False, "performance_metrics": False,
            "tie_rule": "Positive first for future constrained polarity argmax",
            "raw_text_exported": False, "scientific_acceptance": "pending_returned_artifact_review"})
        run.write_text("report.md", "# Original frozen SVM — Teke score extraction\n\n"
            "All primary rows retain three decision margins, stable row/mask/clean-text lineage "
            "and historical hard prediction comparison; no raw text exported. Zero hard-label "
            "and score-argmax disagreements required; learned state unchanged. No fit, calibration, "
            "checkpoint substitution or performance metrics. Original results/masks preserved; "
            "historical training-record gaps not retroactively repaired. Review returned artifacts "
            "before other model score extraction and constrained metrics.\n")
    finally:
        run.write_json("warnings.json", [{"category": w.category.__name__, "message": str(w.message)}
                                         for w in captured])
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after,
                                                   "checked_files": len(before)})
        require(before == after, "Protected input changed")
    run.declare_complete(["scores.csv", "input_inventory.json", "estimator_record.json",
                          "reproduction_check.json", "warnings.json", "preservation_check.json",
                          "results.json", "report.md"])


def main():
    output = ROOT / "outputs/external" / RUN_ID
    require(not output.exists(), "Refusing existing score-extraction directory")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_frozen_teke_svm_score_extraction",
                       config={"contract_sha256": CONTRACT_PIN, "preflight_report_sha256": REPORT_PIN,
                               "preflight_inventory_sha256": INVENTORY_PIN, "device": "cpu",
                               "batch_size": BATCH_SIZE, "training": False, "calibration": False,
                               "performance_metrics": False}) as run:
        extract(run)


if __name__ == "__main__":
    main()
