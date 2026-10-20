"""User-run R08 inventory: hashes/CSV lineage/saved scores; no model loading.

No fitting, inference, dataset download, metric calculation or artifact mutation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_teke_external_inference import ROSTER, check_artifact
from src.protocol_constants import WINVOKER_TEST_MASK_PATH, WINVOKER_TEST_MASK_SHA256

RUN_ID = "professor_round1_winvoker_polarity_input_inventory_v1"
LABELS = ("Positive", "Negative", "Notr")
SCORE_COLUMNS = ("logit_positive", "logit_negative", "logit_notr")
PREDICTION_PINS = {
    "berturk": "1b691e2ea9b349dd3729c5427b9cc634ae11322e87711ca3c56f51cb4fda5974",
    "xlm_roberta": "5b92007fc5a7d41431f1b472dcc2ed7f4157ec110d44a2ad9a8db425bfb80558",
    "bilstm": "80ce705a53896905278d7678d1fd067daf5e8ad99654cc8c07e996898e8a7a18",
}
SVM_PIN = "0e381fcc40572297b9e792bc2d5cec8eb224dcfdc712feb75455d23a266677d0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                         allow_nan=False) + "\n", encoding="utf-8")


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def inventory_paths():
    files = {Path(__file__), ROOT / "scripts/preflight_teke_external_inference.py",
             ROOT / "src/protocol_constants.py", WINVOKER_TEST_MASK_PATH,
             ROOT / "paper/artifacts/teke_external_checkpoint_roster_v1.md"}
    for item in ROSTER:
        directory = ROOT / "outputs/experiments" / item["run"]
        files.update(directory / name for name in
                     (item["artifact"], "config.json", "predictions.csv", "metrics.json"))
        if item["slot"] != "linear_svm":
            files.update(directory / name for name in
                         ("artifact_manifest.json", "run_status.json", "checkpoint_metadata.json"))
    return sorted(files)


def inspect(output):
    require(digest(WINVOKER_TEST_MASK_PATH) == WINVOKER_TEST_MASK_SHA256, "Frozen mask changed")
    lineage = list(rows(WINVOKER_TEST_MASK_PATH))
    require(len(lineage) == 88136, "Wrong test support")
    support = {"complete": Counter(), "strict": Counter()}
    sources = {"complete": Counter(), "strict": Counter()}
    seen = set()
    for index, row in enumerate(lineage):
        require(int(row["test_position"]) == index, "Mask ordering changed")
        require(row["source_row_index"] not in seen, "Duplicate lineage row")
        seen.add(row["source_row_index"])
        require(row["label"] == LABELS[int(row["label_id"])], "Mask label order mismatch")
        require(row["strict_included"] in ("True", "False"), "Invalid strict flag")
        for view in ("complete", "strict"):
            if view == "strict" and row["strict_included"] != "True":
                continue
            support[view][row["label"]] += 1
            if row["label"] in LABELS[:2]:
                sources[view][row["dataset"]] += 1
    require(sum(support["strict"].values()) == 87632, "Strict mask changed")
    require(sum(sources["complete"].values()) == 57371 and
            sum(sources["strict"].values()) == 56870, "Polarity supports changed")
    require(set(sources["complete"]) == {"HUMIR", "urun_yorumlari", "magaza_yorumlari", "tweet-pn"},
            "Unexpected gold polarity sources")
    write(output, "lineage_summary.json", {"mask_sha256": WINVOKER_TEST_MASK_SHA256,
        "label_order": LABELS, "supports": support, "polarity_by_source": sources,
        "excluded_sources": {"wiki": "no gold polarity support", "random": "no gold polarity support"}})
    records = []
    for item in ROSTER:
        slot = item["slot"]
        artifact = check_artifact(item)  # Hash/static metadata only: no deserialization.
        if slot == "linear_svm":
            require(artifact["sha256"] == SVM_PIN, "Historical frozen SVM bundle changed")
        directory = ROOT / "outputs/experiments" / item["run"]
        prediction = directory / "predictions.csv"
        observed_sha = digest(prediction)
        if slot != "linear_svm":
            require(observed_sha == PREDICTION_PINS[slot], "Saved neural predictions changed")
            manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
            entries = [e for e in manifest["files"] if e["relative_to_run"] == "predictions.csv"]
            require(len(entries) == 1 and entries[0]["sha256"] == observed_sha and
                    entries[0]["size_bytes"] == prediction.stat().st_size, "Prediction manifest mismatch")
        count = ties = disagreements = 0
        for index, row in enumerate(rows(prediction)):
            require(index < len(lineage), f"Extra {slot} row")
            ref = lineage[index]
            require(int(row["true_id"]) == int(ref["label_id"]) and
                    row["true_label"] == ref["label"], f"{slot} gold mismatch at {index}")
            require(hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == ref["cleaned_text_sha256"],
                    f"{slot} cleaned text mismatch at {index}")
            require(int(row["predicted_id"]) in range(3) and
                    row["predicted_label"] == LABELS[int(row["predicted_id"])], "Prediction label mismatch")
            if slot != "linear_svm":
                for field in ("test_position", "source_row_index", "cleaned_text_sha256",
                              "strict_included", "strict_unique_included"):
                    require(row[field] == ref[field], f"{slot} {field} mismatch at {index}")
                require(row["source"] == ref["dataset"] and row["sample_id"] ==
                        f"winvoker-train-4bd9103ad5f2-{int(ref['source_row_index']):06d}", "Sample/source mismatch")
                values = [float(row[name]) for name in SCORE_COLUMNS]
                require(all(math.isfinite(v) for v in values), "Missing/nonfinite logit")
                disagreements += int(max(range(3), key=lambda k: values[k]) != int(row["predicted_id"]))
                ties += int(ref["label"] in LABELS[:2] and values[0] == values[1])
            else:
                require(not any(name in row for name in SCORE_COLUMNS), "Unexpected SVM score schema; inspect before reuse")
            count += 1
        require(count == 88136, f"Incomplete {slot} rows")
        require(disagreements == 0, f"{slot} saved-score argmax differs from stored labels: {disagreements}")
        records.append({**artifact, "predictions_sha256": observed_sha, "rows": count,
            "saved_score_kind": "neural_logits" if slot != "linear_svm" else "absent",
            "three_class_argmax_disagreements": disagreements, "gold_polarity_score_ties": ties,
            "next_action": "reuse_saved_logits" if slot != "linear_svm" else "CPU_frozen_bundle_decision_function_extraction_no_fit",
            "historical_limitations": "SVM lacks training manifest and stable prediction IDs; positional gold/text alignment checked" if slot == "linear_svm" else "checkpoint-specific, no external seed variance"})
        write(output, "score_inventory.json", records)
        print(f"Validated {slot}: {count} aligned rows; scores {'missing' if slot == 'linear_svm' else 'available'}", flush=True)
    write(output, "results.json", {"status": "inventory_passed", "models": 4,
        "metric_calculation": False, "training": False, "inference": False,
        "polarity_tie_rule": "Positive first in fixed Positive/Negative score order",
        "next": "prepare frozen SVM CPU score extraction and constrained two-score metrics",
        "public_package_inclusion": False})


def main():
    output = ROOT / "outputs/audits" / RUN_ID
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"run_id": RUN_ID, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "training": False, "inference": False, "network": False, "error": None}
    write(output, "run_status.json", status)
    before = None
    failure = None
    try:
        write(output, "environment.json", {"python": sys.version, "platform": platform.platform()})
        write(output, "config.json", {"roster": ROSTER, "prediction_pins": PREDICTION_PINS,
              "svm_pin": SVM_PIN, "mask_pin": WINVOKER_TEST_MASK_SHA256, "score_order": LABELS})
        before = {p.relative_to(ROOT).as_posix(): digest(p) for p in inventory_paths()}
        write(output, "input_inventory.json", before)
        inspect(output)
    except Exception as exc:
        failure = exc
        status["error"] = traceback.format_exc()
        (output / "error.txt").write_text(status["error"], encoding="utf-8")
    finally:
        if before is not None:
            try:
                after = {p.relative_to(ROOT).as_posix(): digest(p) for p in inventory_paths()}
                write(output, "preservation_check.json", {"inputs_unchanged": before == after,
                                                          "checked_files": len(before)})
                require(before == after, "Inventory inputs changed")
            except Exception as exc:
                failure = failure or exc
                status["error"] = status["error"] or traceback.format_exc()
        status.update(status="failed" if failure else "complete", elapsed_seconds=time.perf_counter() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write(output, "run_status.json", status)
        write(output, "artifact_manifest.json", {"run_id": RUN_ID, "status": status["status"],
            "files": [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": digest(p)}
                      for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"]})
    if failure:
        raise failure
    print(f"Run completed: {RUN_ID}", flush=True)


if __name__ == "__main__":
    main()
