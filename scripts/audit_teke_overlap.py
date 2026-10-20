"""CPU-only integrity, blind-review, and Winvoker exact-overlap audit for Teke XLSX.

Reads the original ZIP in memory; never trains, loads a model, or predicts labels.
The source XLSX has no official evaluation split, so this script creates none.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import re
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_text
from src.protocol_constants import (
    WINVOKER_ASSIGNMENTS_PATH,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
)

SOURCE = ROOT / "data/external/teke_trendyol_original.zip"
INPUT_INVENTORY = ROOT / "outputs/audits/teke_source_inventory_v1/inventory.json"
INPUT_MANIFEST = ROOT / "outputs/audits/teke_source_inventory_v1/manifest.json"
OUTPUT = ROOT / "outputs/audits/teke_overlap_v1"
EXPECTED_ARCHIVE_SHA256 = "e2f1bc8ab44a6b23785c87c557a5df0d9cf5886d4198531807bcd626bbefb7b2"
EXPECTED_WORKBOOK_SHA256 = "d502ef0f0b12ea557a816acc3562596d2a80ff60e826d1d6ce832d9aac480de2"
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
SHEET_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
CELL_COLUMN = re.compile(r"^([A-Z]+)")
SCHEMA = "teke-overlap-v1"
LABEL_MAP = {"pozitif": "Positive", "negatif": "Negative", "notr": "Notr"}
EXPECTED_COUNTS = Counter({"Positive": 60109, "Negative": 11867, "Notr": 1416})
SPLITS = ("train", "validation", "test")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell_text(cell: ET.Element, strings: list[str]) -> str:
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{MAIN}}}t"))
    value = cell.find(f"{{{MAIN}}}v")
    if value is None or value.text is None:
        return ""
    return strings[int(value.text)] if cell.get("t") == "s" else value.text


def load_teke_rows() -> tuple[list[dict], dict]:
    inventory_bytes = INPUT_INVENTORY.read_bytes()
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("inventory.json") != digest(inventory_bytes):
        raise ValueError("Inventory manifest hash mismatch")
    inventory = json.loads(inventory_bytes)
    if inventory.get("schema_version") != "teke-source-inventory-v1":
        raise ValueError("Unexpected source inventory schema")
    if len(inventory["sheets"]) != 1 or inventory["sheets"][0]["data_rows"] != 73392:
        raise ValueError("Inventory workbook structure changed")
    original = SOURCE.read_bytes()
    if digest(original) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original Teke ZIP hash changed")
    with zipfile.ZipFile(io.BytesIO(original)) as outer:
        members = [info for info in outer.infolist() if not info.is_dir()]
        if len(members) != 1 or members[0].filename != inventory["source"]["workbook_member"]:
            raise ValueError("Original ZIP member changed")
        workbook_bytes = outer.read(members[0])
    if digest(workbook_bytes) != EXPECTED_WORKBOOK_SHA256:
        raise ValueError("Original Teke XLSX hash changed")
    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            with book.open("xl/sharedStrings.xml") as shared:
                for _, element in ET.iterparse(shared, events=("end",)):
                    if element.tag == f"{{{MAIN}}}si":
                        strings.append("".join(node.text or "" for node in element.iter(f"{{{MAIN}}}t")))
                        element.clear()
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.get("Id"): item.get("Target")
            for item in relationships.findall(f"{{{PKG_REL}}}Relationship")
            if item.get("Type") == SHEET_REL
        }
        sheet = workbook.find(f".//{{{MAIN}}}sheet")
        target = targets[sheet.get(f"{{{REL}}}id")]
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        if path != inventory["sheets"][0]["sheet_member"]:
            raise ValueError("Worksheet path differs from inventory")
        rows = []
        with book.open(path) as source:
            for _, element in ET.iterparse(source, events=("end",)):
                if element.tag != f"{{{MAIN}}}row":
                    continue
                excel_row = int(element.get("r", "0"))
                values = {}
                for cell in element.findall(f"{{{MAIN}}}c"):
                    match = CELL_COLUMN.match(cell.get("r", ""))
                    if match:
                        values[match.group(1)] = cell_text(cell, strings)
                if excel_row == 1:
                    if values != {"A": "yorumlar", "B": "duygu"}:
                        raise ValueError(f"Unexpected XLSX header: {values}")
                else:
                    raw = values.get("A", "")
                    label_raw = values.get("B", "")
                    label_key = label_raw.strip().lower()
                    if label_key not in LABEL_MAP:
                        raise ValueError(f"Unknown label on Excel row {excel_row}: {label_raw!r}")
                    rows.append({
                        "sample_id": f"teke:Sayfa1:row:{excel_row}",
                        "excel_row": excel_row,
                        "raw_text": raw,
                        "clean_text": clean_text(raw),
                        "label": LABEL_MAP[label_key],
                        "label_raw": label_raw,
                    })
                element.clear()
    if len(rows) != 73392 or Counter(row["label"] for row in rows) != EXPECTED_COUNTS:
        raise ValueError("Teke row count or normalized label counts differ from inventory")
    if [row["excel_row"] for row in rows] != list(range(2, 73394)):
        raise ValueError("Teke worksheet rows are not the expected contiguous lineage")
    return rows, inventory


def load_winvoker_text_sets() -> tuple[dict, dict]:
    if file_digest(WINVOKER_ASSIGNMENTS_PATH) != WINVOKER_ASSIGNMENTS_SHA256:
        raise ValueError("Pinned Winvoker split assignment hash mismatch")
    assignments = pd.read_csv(WINVOKER_ASSIGNMENTS_PATH, usecols=["split", "source_row_index"])
    indexes = pd.to_numeric(assignments["source_row_index"], errors="raise").astype(int)
    if (
        len(assignments) != 440679
        or indexes.nunique() != 440679
        or indexes.min() != 0
        or indexes.max() != 440678
        or set(assignments["split"]) != set(SPLITS)
    ):
        raise ValueError("Historical Winvoker split assignments are not a full partition")
    split_by_index = dict(zip(indexes, assignments["split"]))
    print("Loading pinned Winvoker text only; no model weights or predictions.", flush=True)
    dataset = load_dataset(WINVOKER_DATASET_ID, revision=WINVOKER_DATASET_REVISION)
    if len(dataset["train"]) != 440679:
        raise ValueError("Pinned Winvoker train row count changed")
    raw_sets = {split: set() for split in SPLITS}
    clean_sets = {split: set() for split in SPLITS}
    for index, value in enumerate(dataset["train"]["text"]):
        raw = "" if value is None else str(value)
        split = split_by_index[index]
        raw_sets[split].add(raw)
        clean_sets[split].add(clean_text(raw))
    raw_sets["publisher_test"] = set()
    clean_sets["publisher_test"] = set()
    if "test" in dataset:
        for value in dataset["test"]["text"]:
            raw = "" if value is None else str(value)
            raw_sets["publisher_test"].add(raw)
            clean_sets["publisher_test"].add(clean_text(raw))
    return raw_sets, clean_sets


def blind_sample(rows: list[dict], count: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda row: digest(f"{salt}\0{row['sample_id']}".encode("utf-8")))[:count]


def review_row(row: dict, reason: str) -> dict:
    return {
        "sample_id": row["sample_id"],
        "excel_row": row["excel_row"],
        "published_label": row["label"],
        "review_text": row["raw_text"],
        "selection_reason": reason,
        "human_label": "",
        "human_notes": "",
    }


def audit(output: Path) -> None:
    rows, inventory = load_teke_rows()
    raw_sets, clean_sets = load_winvoker_text_sets()
    raw_groups: dict[str, set[str]] = defaultdict(set)
    clean_groups: dict[str, set[str]] = defaultdict(set)
    raw_counts = Counter()
    clean_counts = Counter()
    for row in rows:
        raw_groups[row["raw_text"]].add(row["label"])
        clean_groups[row["clean_text"]].add(row["label"])
        raw_counts[row["raw_text"]] += 1
        clean_counts[row["clean_text"]] += 1
    flags = [f"{kind}_overlap_{split}" for kind in ("raw", "clean") for split in (*SPLITS, "publisher_test")]
    flags += ["clean_overlap_development", "empty_after_cleaning", "conflicting_raw_label", "conflicting_clean_label"]
    lineage = []
    issues = Counter()
    by_label = {label: Counter() for label in LABEL_MAP.values()}
    for row in rows:
        raw = row["raw_text"]
        cleaned = row["clean_text"]
        record = {
            "sample_id": row["sample_id"],
            "excel_row": row["excel_row"],
            "label": row["label"],
            "label_raw": row["label_raw"],
            "raw_text_sha256": digest(raw.encode("utf-8")),
            "clean_text_sha256": digest(cleaned.encode("utf-8")),
            "duplicate_raw_text": raw_counts[raw] > 1,
            "duplicate_clean_text": clean_counts[cleaned] > 1,
            "unicode_replacement": "\ufffd" in raw,
            "empty_raw_text": not raw.strip(),
        }
        for split in (*SPLITS, "publisher_test"):
            record[f"raw_overlap_{split}"] = raw in raw_sets[split]
            record[f"clean_overlap_{split}"] = bool(cleaned) and cleaned in clean_sets[split]
        record["clean_overlap_development"] = record["clean_overlap_train"] or record["clean_overlap_validation"]
        record["empty_after_cleaning"] = not cleaned
        record["conflicting_raw_label"] = len(raw_groups[raw]) > 1
        record["conflicting_clean_label"] = len(clean_groups[cleaned]) > 1
        lineage.append(record)
        for key in ["duplicate_raw_text", "duplicate_clean_text", "unicode_replacement", "empty_raw_text", *flags]:
            issues[key] += int(record[key])
            by_label[row["label"]][key] += int(record[key])
    fields = [
        "sample_id", "excel_row", "label", "label_raw", "raw_text_sha256", "clean_text_sha256",
        "duplicate_raw_text", "duplicate_clean_text", "unicode_replacement", "empty_raw_text", *flags,
    ]
    write_csv(output / "row_lineage_and_overlap.csv", lineage, fields)
    review_fields = ["sample_id", "excel_row", "published_label", "review_text", "selection_reason", "human_label", "human_notes"]
    random_review = []
    targeted_review = []
    for label in ("Positive", "Negative", "Notr"):
        eligible = [row for row, record in zip(rows, lineage) if row["label"] == label and not record["empty_after_cleaning"]]
        random_review.extend(review_row(row, "class-stratified deterministic random-key sample") for row in blind_sample(eligible, 10, "teke-random-v1"))
        by_id = {row["sample_id"]: record for row, record in zip(rows, lineage)}
        selected = {}
        for flag, count, reason in (
            ("conflicting_clean_label", 4, "conflicting labels for identical cleaned text"),
            ("clean_overlap_development", 3, "exact cleaned Winvoker development overlap"),
            ("duplicate_clean_text", 3, "duplicate cleaned text"),
        ):
            pool = [row for row in eligible if by_id[row["sample_id"]][flag]]
            for row in blind_sample(pool, count, "teke-targeted-v1:" + reason):
                selected.setdefault(row["sample_id"], (row, reason))
        if len(selected) < 10:
            for row in blind_sample(eligible, 30, "teke-targeted-v1:fill"):
                selected.setdefault(row["sample_id"], (row, "deterministic fill"))
                if len(selected) == 10:
                    break
        targeted_review.extend(review_row(row, reason) for row, reason in list(selected.values())[:10])
    write_csv(output / "label_review_random_30.csv", random_review, review_fields)
    write_csv(output / "label_review_targeted_30.csv", targeted_review, review_fields)
    summary = {
        "schema_version": SCHEMA,
        "completed_utc": now(),
        "scientific_status": "audit_only_not_paper_eligible",
        "source_zip_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_xlsx_sha256": EXPECTED_WORKBOOK_SHA256,
        "source_inventory_sha256": file_digest(INPUT_INVENTORY),
        "audit_source_sha256": file_digest(Path(__file__)),
        "canonical_cleaner_source_sha256": file_digest(ROOT / "src/data_loader.py"),
        "label_mapping": LABEL_MAP,
        "label_counts": dict(EXPECTED_COUNTS),
        "rows": len(rows),
        "distinct_raw_texts": len(raw_counts),
        "distinct_clean_texts": len(clean_counts),
        "conflicting_raw_text_groups": sum(len(labels) > 1 for labels in raw_groups.values()),
        "conflicting_clean_text_groups": sum(len(labels) > 1 for labels in clean_groups.values()),
        "issue_and_overlap_row_counts": dict(issues),
        "issue_and_overlap_row_counts_by_label": {label: dict(values) for label, values in by_label.items()},
        "winvoker_dataset_id": WINVOKER_DATASET_ID,
        "winvoker_revision": WINVOKER_DATASET_REVISION,
        "winvoker_assignment_sha256": WINVOKER_ASSIGNMENTS_SHA256,
        "review_sampling": "10 deterministic SHA-256-ranked nonempty rows per class in each prediction-blind sheet; targeted sheet is not a prevalence sample",
        "not_measured": ["Kaggle license/version proof", "lexical or semantic near-overlap", "human label agreement", "model transfer performance"],
        "inventory_source_url": inventory["source"]["source_url"],
    }
    write_json(output / "audit.json", summary)
    print("Audit complete; review audit.json and blind samples before any model evaluation.", flush=True)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing audit output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = now()
    write_json(OUTPUT / "config.json", {
        "schema_version": SCHEMA,
        "source_zip": str(SOURCE.relative_to(ROOT)),
        "source_inventory": str(INPUT_INVENTORY.relative_to(ROOT)),
        "output_dir": str(OUTPUT.relative_to(ROOT)),
    })
    write_json(OUTPUT / "run_status.json", {"schema_version": SCHEMA, "status": "running", "started_utc": started})
    try:
        audit(OUTPUT)
    except Exception as error:
        write_json(OUTPUT / "run_status.json", {
            "schema_version": SCHEMA, "status": "failed", "started_utc": started,
            "completed_utc": now(), "error_type": type(error).__name__,
            "error_message": str(error), "traceback": traceback.format_exc(),
        })
        raise
    else:
        write_json(OUTPUT / "run_status.json", {
            "schema_version": SCHEMA, "status": "complete", "started_utc": started,
            "completed_utc": now(),
        })
        names = ["config.json", "audit.json", "row_lineage_and_overlap.csv", "label_review_random_30.csv", "label_review_targeted_30.csv"]
        write_json(OUTPUT / "manifest.json", {name: file_digest(OUTPUT / name) for name in names})


if __name__ == "__main__":
    main()
