"""Shared file/lineage gates only: never imports or deserializes models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiment_artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Positive", "Negative", "Notr")
SLOTS = ("berturk", "xlm_roberta", "bilstm", "linear_svm")
CONTRACT = ROOT / "configs/analysis/professor_round1_teke_score_contract_v1.json"
CONTRACT_PIN = "e1346f81cdd152a72a4e496761616eb4dcd816079a25958708305b1b84a11af5"
MASK = ROOT / "outputs/audits/teke_external_final_mask_v1/final_mask.csv"
PINS = {
    "berturk": "99615f453461aa7c47a3ba5068da6829f124a00ebc04412abd756ef117a3126c",
    "xlm_roberta": "4fa7bce960aadfb6f458e1122b1d8bc45c03dce214ff7afa58bf3f23e04663b7",
    "bilstm": "30afd26a87f0242f193091fdf63dbcd63734ffd4966e847aa5484fbbdd1ff84c",
    "linear_svm": "a984c35412238738fbe962d2dffd023ce08c703d12d9e2f537dd3af288f7fe14",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def score_directory(slot):
    require(slot in SLOTS, "Unexpected frozen slot")
    return ROOT / "outputs/external" / f"professor_round1_teke_scores_v1_{slot}"


def manifest_gate(directory):
    status = read_json(directory / "run_status.json")
    require(status["status"] == "complete" and status["error"] is None,
            f"Incomplete run: {directory.name}")
    manifest = read_json(directory / "artifact_manifest.json")
    require(manifest["run_id"] == directory.name and manifest["run_status"] == "complete",
            "Wrong artifact manifest identity")
    seen = set()
    for entry in manifest["files"]:
        name = entry["relative_to_run"]
        require(Path(name).name == name and name not in seen, "Unsafe/duplicate manifest name")
        seen.add(name)
        path = directory / name
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"],
                f"Changed manifested payload: {directory.name}/{name}")


def input_gate():
    require(sha256_file(CONTRACT) == CONTRACT_PIN, "Frozen contract changed")
    contract = read_json(CONTRACT)
    require(sha256_file(MASK) == contract["mask_sha256"], "Frozen external mask changed")
    before = {}
    for slot in SLOTS:
        directory = score_directory(slot)
        manifest_gate(directory)
        require(sha256_file(directory / "scores.csv") == PINS[slot], "Accepted score pin changed")
        result = read_json(directory / "results.json")
        require(result["rows"] == 67964 and result["score_order"] == list(LABELS) and
                result["mask_sha256"] == contract["mask_sha256"] and
                result["source_zip_sha256"] == contract["source_zip_sha256"] and
                result["profile_sha256"] == contract["profile_sha256"] and
                result["ordered_sample_ids_sha256"] == contract["ordered_sample_ids_sha256"],
                "Score result lineage/class order changed")
        cp_field = "bundle_sha256" if slot == "linear_svm" else "checkpoint_sha256"
        require(result[cp_field] == contract["historical"][slot]["checkpoint_sha256"], "Score model changed")
        reproduction = read_json(directory / "reproduction_check.json")
        require(reproduction["rows"] == 67964 and reproduction["historical_prediction_disagreements"] == 0 and
                reproduction["state_unchanged"] and
                read_json(directory / "preservation_check.json")["inputs_unchanged"], "Score acceptance gates changed")
        if slot == "linear_svm":
            require(reproduction["score_argmax_disagreements"] == 0, "SVM score argmax gate changed")
        if slot == "bilstm":
            require(reproduction["vocab_unchanged"], "Frozen vocabulary changed")
        if slot in ("berturk", "xlm_roberta"):
            runtime = read_json(directory / "runtime_gate.json")
            require(runtime["finite_logits"] and runtime["historical_disagreements"] == 0,
                    "Transformer first-batch gate changed")
        if slot == "xlm_roberta":
            require(read_json(directory / "preceding_berturk_preservation_check.json")["inputs_unchanged"],
                    "Preceding BERTurk preservation failed")
        for inventory in ("input_inventory.json", "preceding_berturk_input_inventory.json"):
            if not (directory / inventory).is_file():
                continue
            for name, pin in read_json(directory / inventory).items():
                path = (ROOT / name).resolve()
                require(path.is_relative_to(ROOT.resolve()), "Unsafe input path")
                require(name not in before or before[name] == pin, "Conflicting frozen input pins")
                before[name] = pin
        for path in directory.iterdir():
            if path.is_file():
                name = path.relative_to(ROOT).as_posix()
                pin = sha256_file(path)
                require(name not in before or before[name] == pin, "Protected output pin conflict")
                before[name] = pin
    for name, pin in before.items():
        require(sha256_file(ROOT / name) == pin, f"Recorded input changed: {name}")
    before[Path(__file__).relative_to(ROOT).as_posix()] = sha256_file(Path(__file__))
    return before, contract


def lineage(contract):
    mask = {r["sample_id"]: r for r in csv_rows(MASK) if r["final_primary_included"] == "True"}
    directory = ROOT / "outputs/external/teke_direct_transfer_v1_berturk"
    require(sha256_file(directory / "predictions.csv") ==
            contract["historical"]["berturk"]["payloads"]["predictions.csv"], "Historical reference changed")
    reference = list(csv_rows(directory / "predictions.csv"))
    require(len(mask) == len(reference) == 67964, "Primary reference incomplete")
    seen = set()
    for row in reference:
        sid = row["sample_id"]
        require(sid in mask and sid not in seen, "Duplicate/unmasked reference ID")
        seen.add(sid)
        ref = mask[sid]
        require(row["excel_row"] == ref["excel_row"] and row["true_label"] == ref["label"] and
                row["clean_text_sha256"] == ref["clean_text_sha256"] and
                row["short_text_sensitivity_included"] == ref["short_text_sensitivity_included"] and
                int(row["true_id"]) == LABELS.index(ref["label"]), "Frozen reference/mask join mismatch")
    return reference
