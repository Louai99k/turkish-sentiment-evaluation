"""User-run byte-identical copy of audited R07 PDFs into NEW working assets.

Never edits PDFs, overwrites assets, refreshes delivery or compiles a manuscript.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_artifacts import ExperimentRun, sha256_file
from scripts.audit_professor_figure_pdfs import PDF_PINS

RUN_ID = "professor_round1_figure_promotion_v1"
SOURCE = ROOT / "outputs/figures/professor_round1_figure_repairs_v2"
AUDIT = ROOT / "outputs/audits/professor_round1_figure_pdf_audit_v2"
DESTINATION = ROOT / "paper/revisions/professor_round1/figures"
MAPPING = {
    "training_history.pdf": "berturk_weighted_ce_seed42_training_history.pdf",
    "roc_complete.pdf": "berturk_standard_ce_seed52_roc_complete.pdf",
    "attention_examples.pdf": "berturk_standard_ce_seed52_attention_examples.pdf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(directory):
    require(read_json(directory / "run_status.json")["status"] == "complete", f"Incomplete input: {directory}")
    manifest = read_json(directory / "artifact_manifest.json")
    require(manifest["run_id"] == directory.name and manifest["run_status"] == "complete", "Wrong manifest identity")
    for entry in manifest["files"]:
        path = (directory / entry["relative_to_run"]).resolve()
        require(path.is_relative_to(directory.resolve()), "Manifest path escapes input")
        require(path.is_file() and path.stat().st_size == entry["size_bytes"] and
                sha256_file(path) == entry["sha256"], f"Input manifest mismatch: {path}")


def protected_inventory():
    files = {Path(__file__).resolve(), ROOT / "scripts/audit_professor_figure_pdfs.py",
             ROOT / "paper/revisions/professor_round1/sn-article.tex",
             ROOT / "configs/manuscript/professor_round1_delivery_v2.json",
             ROOT / "paper/artifacts/professor_round1_figure_pdf_audit_review_v2.md"}
    for directory in (SOURCE, AUDIT, ROOT / "paper/overleaf", ROOT / "paper/professor_supplement",
                      ROOT / "paper/revisions/professor_round1/downloaded"):
        require(directory.is_dir(), f"Missing protected input: {directory}")
        files.update(path for path in directory.rglob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(files)}


def promote(run):
    before = protected_inventory()
    run.write_json("input_inventory.json", before)
    records = []
    try:
        verify_manifest(SOURCE)
        verify_manifest(AUDIT)
        summary = read_json(AUDIT / "audit_summary.json")
        require(summary["status"] == "passed" and summary["pdf_text_checks_passed"] and
                summary["fonts_embedded_TrueType_Unicode"], "PDF audit not accepted at technical gate")
        require(read_json(AUDIT / "preservation_check.json")["inputs_unchanged"], "Audit source preservation failed")
        manuscript = (ROOT / "paper/revisions/professor_round1/sn-article.tex").read_text(encoding="utf-8")
        # Preflight every target before any paper asset is copied.
        for source_name, destination_name in MAPPING.items():
            require(sha256_file(SOURCE / source_name) == PDF_PINS[source_name], "Pinned source PDF changed")
            require(not (DESTINATION / destination_name).exists(), f"Refusing existing asset: {destination_name}")
            require("{figures/" + destination_name + "}" in manuscript, "Working source missing promoted reference")
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for source_name, destination_name in MAPPING.items():
            source, target = SOURCE / source_name, DESTINATION / destination_name
            # Exclusive target creation: even an unexpected concurrent file is never overwritten.
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            digest = sha256_file(target)
            require(digest == PDF_PINS[source_name], "Byte-identical copy check failed")
            records.append({"source": source.relative_to(ROOT).as_posix(),
                            "destination": target.relative_to(ROOT).as_posix(),
                            "size_bytes": target.stat().st_size, "sha256": digest,
                            "byte_identical": True})
            run.write_json("promotion_records.json", records)
        run.write_json("results.json", {"status": "promoted_to_working_assets", "copied_pdfs": len(records),
            "pdf_content_modified": False, "delivery_refreshed": False, "compiled_review": "pending",
            "final_delivery_ready": False})
        run.write_text("report.md", "# Audited figure promotion\n\nThree PDFs copied byte-for-byte to new authoritative working-figure paths. No PDF transformation or overwrite, no Overleaf/supplement refresh, no original figure deletion. Separate candidate staging and scaled compiled-page review remain. Partial copies on a failed run are preserved with any available receipts; do not delete or retry blindly.\n")
    finally:
        after = protected_inventory()
        run.write_json("preservation_check.json", {"status": "passed" if before == after else "failed",
            "checked_files": len(before), "protected_inputs_unchanged": before == after,
            "current_delivery_modified": False})
        require(before == after, "Protected source/delivery inputs changed during promotion")
    run.declare_complete(["input_inventory.json", "promotion_records.json", "preservation_check.json",
                          "results.json", "report.md"])


def main():
    output = ROOT / "outputs/audits" / RUN_ID
    require(not output.exists(), "Refusing existing promotion run directory")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_byte_identical_working_figure_copy",
                       config={"source_run": SOURCE.name, "audit_run": AUDIT.name,
                               "pdf_pins": PDF_PINS, "mapping": MAPPING,
                               "delivery_refresh": False, "compilation": False}) as run:
        promote(run)


if __name__ == "__main__":
    main()
