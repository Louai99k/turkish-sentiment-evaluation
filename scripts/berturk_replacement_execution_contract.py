"""Read-only execution identity checks; no model imports or training."""
from __future__ import annotations

from pathlib import Path

from scripts.berturk_replacement_contract import (
    ROOT, MATRIX, PINS, preparation_gate, canonical_sha, read, require, sha,
)

RUNTIME = ROOT / "outputs/audits/professor_round1_berturk_replacement_runtime_v1"
EXECUTION_ID = "professor_round1_berturk_replacement_execution_v1"
ACCEPTANCE = ROOT / "configs/final/berturk_replacement_execution_acceptance_v1.json"
SOURCE_FILES = ("scripts/run_berturk_provenance_replacement.py",
                "scripts/berturk_replacement_execution_contract.py",
                "scripts/audit_berturk_replacement_execution.py")
RUNTIME_PINS = {
    "results.json": "96090b5a006587dd52dcf5755195e7c5299e87d7978cff1cd69779cb3e33bcea",
    "runtime_record.json": "99b8c6dbe9eab932e060fed7c0d0040d11af129bc2d33fb9026ef52efa15e15a",
    "input_inventory.json": "24ba5bce9b3f85ccfed9d820dd70cdb330a3aaf8d38a291a3185c0e74a1185ee",
    "logit_comparison.json": "97a2aadb7c5619d519dc7382aaffb111fd5b3cf5b75ed77144622c15178708eb",
}


def manifest_gate(directory):
    status = read(directory / "run_status.json")
    require(status["status"] == "complete" and status["error"] is None, "Required gate incomplete")
    manifest = read(directory / "artifact_manifest.json")
    require(manifest["run_id"] == directory.name and
            manifest.get("run_status", manifest.get("status")) == "complete", "Gate manifest identity differs")
    seen = set()
    for entry in manifest["files"]:
        name = entry.get("relative_to_run", entry.get("name"))
        require(Path(name).name == name and name not in seen, "Unsafe/duplicate gate payload")
        seen.add(name)
        path = directory / name
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"], "Gate payload changed")


def source_identity():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def scientific_gate():
    matrix, configs, before, cache = preparation_gate()
    manifest_gate(RUNTIME)
    for name, pin in RUNTIME_PINS.items():
        require(sha(RUNTIME / name) == pin, "Accepted runtime pin changed")
    comparison, record = read(RUNTIME / "logit_comparison.json"), read(RUNTIME / "runtime_record.json")
    require(comparison["allclose"] and comparison["exact_argmax_agreement"] and
            record["model_states_unchanged"] and record["gradients_absent"] and
            record["optimizer_state_entries"] == 0, "Runtime scientific gates failed")
    preservation = read(RUNTIME / "preservation_check.json")
    require(preservation["inputs_unchanged"] and preservation["cache_files_unchanged"], "Runtime preservation failed")
    for name, pin in read(RUNTIME / "input_inventory.json").items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha(path) == pin, "Recorded runtime input changed")
        before[name] = pin
    for path in RUNTIME.iterdir():
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    sources = source_identity()
    before.update(sources)
    return matrix, configs, before, cache, sources


def execution_acceptance_gate(sources):
    require(ACCEPTANCE.is_file(), "Execution audit acceptance record absent; training blocked")
    acceptance = read(ACCEPTANCE)
    require(acceptance["status"] == "accepted_for_authorized_six_fit_execution" and
            acceptance["matrix_sha256"] == PINS["matrix"] and
            acceptance["execution_audit_id"] == EXECUTION_ID and acceptance["source_hashes"] == sources,
            "Execution acceptance/source identity differs")
    directory = ROOT / "outputs/audits" / EXECUTION_ID
    manifest_gate(directory)
    report = read(directory / "report.json")
    preservation = read(directory / "preservation_check.json")
    require(sha(directory / "report.json") == acceptance["report_sha256"] and
            report["source_hashes"] == sources and
            report["status"] == "guard_audit_passed_pending_acceptance_record" and
            all(item["passed"] for item in report["checks"]) and
            preservation["inputs_unchanged"] and preservation["cache_files_unchanged"],
            "Execution report/preservation differs")
    return acceptance, directory
