"""XLM-R frozen inference using accepted runner; additionally protect BERTurk outputs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_professor_teke_transformer_scores import extract
from scripts.extract_professor_teke_svm_scores import CONTRACT_PIN, read_json, require
from src.experiment_artifacts import ExperimentRun, sha256_file

RUN_ID = "professor_round1_teke_scores_v1_xlm_roberta"
BERTURK = ROOT / "outputs/external/professor_round1_teke_scores_v1_berturk"
BERTURK_SCORE_PIN = "99615f453461aa7c47a3ba5068da6829f124a00ebc04412abd756ef117a3126c"


def previous_gate():
    status = read_json(BERTURK / "run_status.json")
    require(status["status"] == "complete" and status["error"] is None and
            status["run_id"] == BERTURK.name, "Accepted BERTurk score run incomplete")
    require(sha256_file(BERTURK / "scores.csv") == BERTURK_SCORE_PIN, "Accepted BERTurk scores changed")
    check = read_json(BERTURK / "reproduction_check.json")
    runtime = read_json(BERTURK / "runtime_gate.json")
    require(check["rows"] == 67964 and check["historical_prediction_disagreements"] == 0 and
            check["state_unchanged"] and runtime["historical_disagreements"] == 0 and
            runtime["finite_logits"] and read_json(BERTURK / "preservation_check.json")["inputs_unchanged"],
            "Accepted BERTurk reproduction/preservation gates changed")
    manifest = read_json(BERTURK / "artifact_manifest.json")
    require(manifest["run_id"] == BERTURK.name and manifest["run_status"] == "complete",
            "Wrong BERTurk manifest identity")
    for entry in manifest["files"]:
        name = entry["relative_to_run"]
        require(Path(name).name == name, "Unsafe previous manifest path")
        path = BERTURK / name
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"],
                f"Accepted BERTurk output changed: {name}")
    before = read_json(BERTURK / "input_inventory.json")
    for name, expected in before.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha256_file(path) == expected,
                f"BERTurk recorded input changed: {name}")
    for path in [*BERTURK.iterdir(), Path(__file__).resolve()]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return before


def main():
    require(len(sys.argv) == 1, "This entrypoint runs only the frozen XLM-R slot; no options")
    output = ROOT / "outputs/external" / RUN_ID
    require(not output.exists(), "Refusing existing extraction directory")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cuda_frozen_teke_transformer_score_extraction",
                       config={"model_slot": "xlm_roberta", "contract_sha256": CONTRACT_PIN,
                               "preceding_berturk_score_pin": BERTURK_SCORE_PIN,
                               "device": "cuda", "precision": "float32", "batch_size": 32,
                               "max_length": 128, "pooling": "first_token", "training": False,
                               "calibration": False, "performance_metrics": False}) as run:
        before = previous_gate()
        run.write_json("preceding_berturk_input_inventory.json", before)
        try:
            extract(run, "xlm_roberta")  # Unmodified runner used by accepted BERTurk.
        finally:
            after = {name: sha256_file(ROOT / name) for name in before}
            run.write_json("preceding_berturk_preservation_check.json", {
                "inputs_unchanged": before == after, "checked_files": len(before)})
            require(before == after, "Accepted preceding BERTurk inputs or outputs changed")
        run.declare_complete(["scores.csv", "runtime_gate.json", "input_inventory.json",
            "checkpoint_record.json", "reproduction_check.json", "warnings.json",
            "preservation_check.json", "results.json", "report.md",
            "preceding_berturk_input_inventory.json", "preceding_berturk_preservation_check.json"])


if __name__ == "__main__":
    main()
