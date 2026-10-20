"""User-run, read-only-source Poppler audit of the accepted R07 v2 candidates.

No model execution, figure regeneration, manuscript installation or compilation.
PDF previews are diagnostic outputs, not replacements for the source PDFs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_artifacts import ExperimentRun, sha256_file

SOURCE = ROOT / "outputs/figures/professor_round1_figure_repairs_v2"
RUN_ID = "professor_round1_figure_pdf_audit_v1"
RECOVERY_RUN_ID = "professor_round1_figure_pdf_audit_v2"
# Pins transcribed from the completed user-run v2 artifact manifest.
PDF_PINS = {
    "attention_examples.pdf": "b2e83406a79ce84d48d632fb1e7cf2cfa0ec3d584cf4d2b3e1a0fb00d012e954",
    "font_probe.pdf": "2f638827846c2772ede3723ff684b2576e9381ffcbc3ee847d7a71e84a67a56b",
    "roc_complete.pdf": "5d3917eb968ded67bc97b9d3c71fe35ecf3c4eedb950d3f7cf2fe4a5bfc0608a",
    "training_history.pdf": "3c741ba3c5a4b06bbf85cd2e1138641c7942a094562c2ea4dcdfac622b1a2cc1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value):
    """Ignore extraction whitespace only; never strip or fold Turkish glyphs."""
    return "".join(value.split())


def snapshot(paths):
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def audit(run):
    paths = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    paths += [Path(__file__).resolve(), ROOT / "configs/analysis/professor_round1_figure_repairs_v2.json"]
    before = snapshot(paths)
    run.write_json("input_inventory.json", before)
    invocations = []

    def invoke(command, stem):
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="strict", timeout=120)
        run.write_text(f"tools/{stem}.stdout.txt", completed.stdout)
        run.write_text(f"tools/{stem}.stderr.txt", completed.stderr)
        invocations.append({"command": command, "returncode": completed.returncode,
                            "stdout": f"tools/{stem}.stdout.txt", "stderr": f"tools/{stem}.stderr.txt"})
        run.write_json("tool_invocations.json", invocations)
        require(completed.returncode == 0, f"Poppler command failed: {command}; see tools/{stem}.stderr.txt")
        return completed.stdout

    try:
        require(read_json(SOURCE / "run_status.json")["status"] == "complete", "V2 source not complete")
        manifest = read_json(SOURCE / "artifact_manifest.json")
        require(manifest["run_id"] == "professor_round1_figure_repairs_v2" and
                manifest["run_status"] == "complete", "Wrong source manifest identity")
        for entry in manifest["files"]:
            path = (SOURCE / entry["relative_to_run"]).resolve()
            require(path.is_relative_to(SOURCE.resolve()), "Manifest path escapes source")
            require(path.is_file() and path.stat().st_size == entry["size_bytes"] and
                    sha256_file(path) == entry["sha256"], f"Manifest mismatch: {path.name}")
        for name, digest in PDF_PINS.items():
            require(sha256_file(SOURCE / name) == digest, f"Pinned PDF changed: {name}")
        layout = read_json(SOURCE / "attention_layout_check.json")
        require(layout["main_title_vs_first_heading"] == "passed" and
                layout["footer_vs_last_token_labels"] == "passed", "Source collision checks failed")
        with (SOURCE / "attention_display_mapping.csv").open(encoding="utf-8", newline="") as handle:
            mapping = list(csv.DictReader(handle))
        counts = {}
        positions = {}
        for row in mapping:
            sample = row["sample_id"]
            counts[sample] = counts.get(sample, 0) + 1
            positions.setdefault(sample, []).append(int(row["token_position"]))
        require(len(mapping) == 81 and list(counts.values()) == [22, 42, 17], "Token counts changed")
        require(all(value == list(range(len(value))) for value in positions.values()), "Token order changed")
        combining = [row for row in mapping if row["token"] == "##\u0307"]
        require(len(combining) == 2 and all(row["display_token"] == "##U+0307" and
                row["token_id"] == "1305" for row in combining), "Combining-dot display lineage changed")
        tools = {name: shutil.which(name) for name in ("pdffonts", "pdftotext", "pdftoppm")}
        run.write_json("tool_availability.json", tools)
        require(all(tools.values()), "Poppler tools missing; install poppler-utils in the user container first")
        versions = {}
        for name, executable in tools.items():
            invoke([executable, "-v"], f"{name}_version")
            versions[name] = {"executable": executable, "sha256": sha256_file(Path(executable)),
                              "version_stdout": f"tools/{name}_version.stdout.txt",
                              "version_stderr": f"tools/{name}_version.stderr.txt"}
        run.write_json("tool_versions.json", versions)
        font_records, text_checks, previews = {}, {}, []
        for name in PDF_PINS:
            stem = Path(name).stem
            pdf = SOURCE / name
            font_text = invoke([tools["pdffonts"], str(pdf)], f"{stem}_fonts")
            rows = []
            # Last five columns are emb/sub/uni/object/ID; type may contain spaces.
            for line in font_text.splitlines()[2:]:
                fields = line.split()
                if not fields:
                    continue
                require(len(fields) >= 8 and fields[-2].isdigit() and fields[-1].isdigit(),
                        f"Unrecognized pdffonts row: {line}")
                row = {"name": fields[0], "details": " ".join(fields[1:-5]),
                       "embedded": fields[-5], "subset": fields[-4], "unicode": fields[-3],
                       "object_id": fields[-2:]}
                rows.append(row)
                require(row["embedded"] == "yes" and row["unicode"] == "yes" and
                        "TrueType" in row["details"], f"Unembedded/non-Unicode/non-TrueType font in {name}: {line}")
            require(rows, f"No fonts found in {name}")
            font_records[name] = rows
            run.write_json("font_records.json", font_records)
            # -raw avoids reading-order rearrangement of rotated attention tokens.
            extracted = invoke([tools["pdftotext"], "-raw", "-enc", "UTF-8", str(pdf), "-"], f"{stem}_text")
            run.write_text(f"extracted_text/{stem}.txt", extracted)
            text = compact(extracted)
            require(bool(text) and "\ufffd" not in text, f"Empty/replacement-character PDF text: {name}")
            if stem == "font_probe":
                checks = {f"glyph_{char}": char in extracted for char in "ğışĞİŞöüç"}
            elif stem == "attention_examples":
                checks = {"all_display_tokens_present": all(compact(row["display_token"]) in text for row in mapping),
                          "combining_dot_markers_present": text.count("##U+0307") >= 2,
                          "Turkish_glyphs_present": all(char in extracted for char in "ğış"),
                          "interpretation_caveat": compact("Attention is not causal importance.") in text}
            elif stem == "roc_complete":
                checks = {"Neutral_precision": compact("Neutral (Notr): AUC=0.99997") in text,
                          "loss_and_seed_label": compact("standard CE | training seed 52") in text}
            else:
                history = read_json(SOURCE / "history_plot_data.json")
                checks = {"source_integer_epochs": history["epochs"] == [1, 2, 3],
                          "source_selected_epoch": history["selected_epoch"] == 2,
                          "English_epoch_label": "Epoch" in extracted,
                          "loss_label": "weightedCE" in text}
            text_checks[name] = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                                "scope": "Text presence, not geometric order or compiled-page legibility"}
            run.write_json("pdf_text_checks.json", text_checks)
            require(all(checks.values()), f"PDF text checks failed: {name}; see pdf_text_checks.json")
            prefix = run.path(f"pdf_previews/{stem}")
            prefix.parent.mkdir(parents=True, exist_ok=True)
            invoke([tools["pdftoppm"], "-f", "1", "-l", "1", "-r", "100", "-singlefile", "-png",
                    str(pdf), str(prefix)], f"{stem}_preview")
            preview = prefix.with_suffix(".png")
            require(preview.is_file() and preview.stat().st_size > 0, f"Missing PDF preview: {name}")
            previews.append(preview.relative_to(run.output_dir).as_posix())
        run.write_json("audit_summary.json", {"status": "passed", "pdf_count": 4,
            "source_manifest_and_pdf_pins_verified": True, "fonts_embedded_TrueType_Unicode": True,
            "pdf_text_checks_passed": True, "pdf_previews": previews,
            "pdf_preview_visual_review": "pending", "compiled_page_review": "pending",
            "installed_to_paper": False, "final_delivery_ready": False})
        run.write_text("report.md", "# R07 v2 PDF audit\n\nSource manifest/PDF pins, embedded TrueType fonts, Unicode maps and extracted text checked with recorded Poppler versions. Four previews rasterized from the actual PDFs. Token mapping/count/order lineage checked; extraction presence does not establish visual order or legibility.\n\nReview previews, then separately inspect the scaled compiled manuscript. No source PDF/figure/manuscript replacement, regeneration, inference, training or compilation. Do not treat this audit alone as final delivery approval.\n")
    finally:
        after = snapshot(paths)
        run.write_json("preservation_check.json", {"status": "passed" if before == after else "failed",
            "checked_files": len(before), "inputs_unchanged": before == after, "installed_to_paper": False})
        require(before == after, "Source files changed during PDF audit")
    run.declare_complete(["input_inventory.json", "tool_availability.json", "tool_versions.json",
        "tool_invocations.json", "font_records.json", "pdf_text_checks.json", "preservation_check.json",
        "audit_summary.json", "report.md", *previews])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=(RUN_ID, RECOVERY_RUN_ID), default=RUN_ID)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    expected_output = ROOT / "outputs/audits" / args.run_id
    output = (args.output_dir or expected_output).resolve()
    require(output == expected_output.resolve(), "Use the audit output directory matching --run-id")
    require(not output.exists(), "Refusing to overwrite an existing audit directory")
    with ExperimentRun(output, run_id=args.run_id, run_kind="cpu_saved_figure_pdf_audit",
                       config={"source_run_id": SOURCE.name, "pdf_sha256": PDF_PINS,
                               "prior_failed_attempt": RUN_ID if args.run_id == RECOVERY_RUN_ID else None,
                               "training": False, "inference": False, "install_to_paper": False}) as run:
        audit(run)


if __name__ == "__main__":
    main()
