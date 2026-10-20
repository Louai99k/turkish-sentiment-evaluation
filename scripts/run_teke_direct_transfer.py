"""Evaluate one frozen Winvoker model directly on the fixed Teke mask.

One slot per invocation. Never fits, tunes, changes labels, or selects a model
from Teke outcomes. The first invocation should be the CPU-only Linear SVM so
the external row lineage and artifact format can be inspected before GPU use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audit_teke_overlap import load_teke_rows
from scripts.preflight_teke_external_inference import (
    MASK_DIR,
    MASK_SHA,
    OUT as PREFLIGHT_DIR,
    PROFILE_SHA,
    ROSTER,
    SOURCE_SHA,
    sha256,
)
from src.data_loader import clean_text
from src.preprocessing import apply_preprocessing_profile, load_preprocessing_profile
from src.protocol_constants import ID_TO_LABEL, LABEL_TO_ID
from src.protocol_metrics import classification_metrics


EXPECTED_PREFLIGHT_REPORT_SHA = "786f88c8ca4f6de78fba4a61a183dc98511b7bd60714fa20a7ba390dd7c22b80"
EXPECTED_SVM_SHA = "0e381fcc40572297b9e792bc2d5cec8eb224dcfdc712feb75455d23a266677d0"
EXPECTED_SUPPORT = {"Positive": 56429, "Negative": 10777, "Notr": 758}
EXPECTED_SHORT_SUPPORT = {"Positive": 55900, "Negative": 10639, "Notr": 684}
PROFILE = ROOT / "configs/preprocessing/winvoker_canonical_v1.json"
OUTPUT_ROOT = ROOT / "outputs/external"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_preflight(slot: str) -> tuple[dict, dict]:
    require(read_json(PREFLIGHT_DIR / "run_status.json")["status"] == "complete", "Artifact preflight incomplete")
    manifest = read_json(PREFLIGHT_DIR / "manifest.json")
    require(manifest["report.json"] == EXPECTED_PREFLIGHT_REPORT_SHA, "Unexpected preflight manifest")
    require(sha256(PREFLIGHT_DIR / "report.json") == EXPECTED_PREFLIGHT_REPORT_SHA,
            "Artifact preflight report changed")
    report = read_json(PREFLIGHT_DIR / "report.json")
    require(report["scientific_status"] == "pre_inference_no_model_loaded", "Unexpected preflight status")
    require(report["mask"]["source_zip_sha256"] == SOURCE_SHA and
            report["mask"]["final_mask_sha256"] == MASK_SHA, "Preflight mask/source changed")
    item = next((entry for entry in ROSTER if entry["slot"] == slot), None)
    require(item is not None, f"Unknown frozen slot: {slot}")
    roster_entry = next((entry for entry in report["roster"] if entry["slot"] == slot), None)
    require(roster_entry is not None and roster_entry["source_run"] == item["run"],
            "Preflight roster mismatch")
    expected_hash = item["expected_sha256"] or EXPECTED_SVM_SHA
    require(roster_entry["sha256"] == expected_hash, f"{slot} preflight artifact hash changed")
    artifact = ROOT / roster_entry["artifact"]
    require(sha256(artifact) == expected_hash, f"{slot} artifact changed after preflight")
    require(sha256(MASK_DIR / "final_mask.csv") == MASK_SHA, "Frozen Teke mask changed")
    require(sha256(PROFILE) == PROFILE_SHA, "Canonical preprocessing profile changed")
    return roster_entry, item


def load_masked_rows() -> tuple[list[dict], dict]:
    flags = {}
    with (MASK_DIR / "final_mask.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["final_primary_included"] != "True":
                continue
            require(row["sample_id"] not in flags, "Duplicate primary sample ID")
            flags[row["sample_id"]] = row
    require(len(flags) == 67964, "Primary mask row count changed")
    _, profile = load_preprocessing_profile(PROFILE)
    original_rows, _ = load_teke_rows()
    selected = []
    for source_row in original_rows:
        mask = flags.pop(source_row["sample_id"], None)
        if mask is None:
            continue
        require(mask["excel_row"] == str(source_row["excel_row"]) and
                mask["label"] == source_row["label"], "Teke row/label lineage mismatch")
        cleaned = apply_preprocessing_profile(source_row["raw_text"], profile)
        require(cleaned == source_row["clean_text"] == clean_text(source_row["raw_text"]),
                f"Canonical cleaning mismatch: {source_row['sample_id']}")
        clean_sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        require(clean_sha == mask["clean_text_sha256"], "Teke cleaned-text hash mismatch")
        selected.append({
            "sample_id": source_row["sample_id"],
            "excel_row": source_row["excel_row"],
            "text": cleaned,
            "clean_text_sha256": clean_sha,
            "true_label": source_row["label"],
            "short_text_sensitivity_included": mask["short_text_sensitivity_included"] == "True",
        })
    require(not flags and len(selected) == 67964, "Primary mask/sample join incomplete")
    support = Counter(row["true_label"] for row in selected)
    short = Counter(row["true_label"] for row in selected if row["short_text_sensitivity_included"])
    require(dict(support) == EXPECTED_SUPPORT and dict(short) == EXPECTED_SHORT_SUPPORT,
            "Joined Teke support changed")
    return selected, {"primary": dict(support), "short_text_sensitivity": dict(short)}


def predict_svm(texts: list[str], artifact: Path) -> np.ndarray:
    # The exact bundle hash was checked before unpickling this trusted local file.
    with artifact.open("rb") as stream:
        bundle = pickle.load(stream)
    require(set(bundle) >= {"vectorizer", "model"}, "SVM bundle format mismatch")
    model = bundle["model"]
    require(set(int(value) for value in model.classes_) == {0, 1, 2}, "SVM class IDs changed")
    matrix = bundle["vectorizer"].transform(texts)
    return np.asarray(model.predict(matrix), dtype=np.int64)


def predict_transformer(slot: str, texts: list[str], artifact: Path, item: dict,
                        device_name: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    from src.model import BERTurkSentimentClassifier
    from src.transformer_model import TransformerSentimentClassifier

    require(device_name == "cuda" and torch.cuda.is_available(),
            "Frozen Transformer transfer requires an available CUDA device")
    device = torch.device("cuda")
    metadata = read_json(ROOT / "outputs/experiments" / item["run"] / "checkpoint_metadata.json")["metadata"]
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer_id"], revision=metadata["tokenizer_revision"], local_files_only=True,
    )
    if slot == "berturk":
        model = BERTurkSentimentClassifier(
            model_name=metadata["model_id"], model_revision=metadata["model_revision"],
            num_classes=3, dropout=float(metadata["dropout"]), local_files_only=True,
        ).to(device)
    else:
        model = TransformerSentimentClassifier(
            model_id=metadata["model_id"], model_revision=metadata["model_revision"],
            pooling=metadata["pooling"], num_classes=3,
            dropout=float(metadata["dropout"]), local_files_only=True,
        ).to(device)
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    require(checkpoint["metadata"]["experiment_id"] == item["run"],
            "Transformer checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], add_special_tokens=True,
                max_length=128, padding="max_length", truncation=True,
                return_attention_mask=True, return_tensors="pt",
            )
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            if (start // batch_size + 1) % 100 == 0:
                print(f"{slot}: {min(start + batch_size, len(texts))}/{len(texts)} rows", flush=True)
    return np.asarray(predictions, dtype=np.int64)


def predict_bilstm(texts: list[str], artifact: Path, item: dict,
                   device_name: str, batch_size: int) -> np.ndarray:
    import torch

    from scripts.run_stage2_neural_baselines import BiLSTMSentimentClassifier, encode_text
    from src.preprocessing import canonical_json_sha256

    require(device_name in ("cpu", "cuda"), "Invalid BiLSTM device")
    require(device_name != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    device = torch.device(device_name)
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    vocab = checkpoint["vocab"]
    metadata = read_json(ROOT / "outputs/experiments" / item["run"] / "checkpoint_metadata.json")
    require(checkpoint["run_identity"]["run_id"] == item["run"] and
            canonical_json_sha256(vocab) == checkpoint["vocab_sha256"] == metadata["vocab_sha256"],
            "BiLSTM checkpoint/vocabulary identity mismatch")
    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab), embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]), num_classes=int(config["num_classes"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = [encode_text(text, vocab, int(config["max_length"]))
                       for text in texts[start:start + batch_size]]
            ids = torch.tensor([row[0] for row in encoded], dtype=torch.long, device=device)
            attention = torch.tensor([row[1] for row in encoded], dtype=torch.long, device=device)
            logits = model(ids, attention)
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            if (start // batch_size + 1) % 100 == 0:
                print(f"bilstm: {min(start + batch_size, len(texts))}/{len(texts)} rows", flush=True)
    return np.asarray(predictions, dtype=np.int64)


def write_outputs(out: Path, slot: str, rows: list[dict], predictions: np.ndarray,
                  support: dict, artifact_info: dict) -> None:
    require(len(predictions) == len(rows) and
            set(int(value) for value in predictions) <= {0, 1, 2}, "Invalid prediction IDs")
    true_ids = [LABEL_TO_ID[row["true_label"]] for row in rows]
    short_indexes = [index for index, row in enumerate(rows) if row["short_text_sensitivity_included"]]
    metrics = {
        "schema_version": "teke-direct-transfer-metrics-v1",
        "model_slot": slot,
        "selection_basis": "Winvoker validation only; frozen before Teke inference",
        "views": {
            "primary": classification_metrics(true_ids, predictions, subset_id="teke_primary"),
            "short_text_sensitivity": classification_metrics(
                [true_ids[index] for index in short_indexes],
                [int(predictions[index]) for index in short_indexes],
                subset_id="teke_short_text_sensitivity",
            ),
        },
    }
    for view, counts in (("primary", support["primary"]),
                         ("short_text_sensitivity", support["short_text_sensitivity"])):
        require({label: metrics["views"][view]["per_class"][label]["support"]
                 for label in EXPECTED_SUPPORT} == counts, f"{view} metric support changed")
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("sample_id", "excel_row", "clean_text_sha256", "true_label", "true_id",
                  "predicted_label", "predicted_id", "short_text_sensitivity_included")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, true_id, predicted_id in zip(rows, true_ids, predictions):
            writer.writerow({
                "sample_id": row["sample_id"], "excel_row": row["excel_row"],
                "clean_text_sha256": row["clean_text_sha256"],
                "true_label": row["true_label"], "true_id": true_id,
                "predicted_label": ID_TO_LABEL[int(predicted_id)], "predicted_id": int(predicted_id),
                "short_text_sensitivity_included": row["short_text_sensitivity_included"],
            })
    write_json(out / "metrics.json", metrics)
    write_json(out / "lineage.json", {
        "schema_version": "teke-direct-transfer-lineage-v1",
        "model_slot": slot,
        "source_zip_sha256": SOURCE_SHA,
        "mask_sha256": MASK_SHA,
        "canonical_profile_sha256": PROFILE_SHA,
        "preflight_report_sha256": EXPECTED_PREFLIGHT_REPORT_SHA,
        "model_artifact": artifact_info,
        "primary_rows": len(rows),
        "short_text_sensitivity_rows": len(short_indexes),
        "ordered_sample_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode("utf-8")).hexdigest(),
        "no_external_fit_or_selection": True,
        "license": "Kaggle page declares CC BY-NC 4.0; raw text intentionally absent from outputs",
    })
    write_json(out / "manifest.json", {
        name: sha256(out / name) for name in ("predictions.csv", "metrics.json", "lineage.json")
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=tuple(item["slot"] for item in ROSTER), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.batch_size <= 256, "Batch size must be 1..256")
    require(args.slot != "linear_svm" or args.device == "cpu", "SVM must run on CPU")
    require(args.slot not in ("berturk", "xlm_roberta") or args.device == "cuda",
            "Transformer slots require CUDA")
    out = OUTPUT_ROOT / f"teke_direct_transfer_v1_{args.slot}"
    out.mkdir(parents=True, exist_ok=False)
    status = {"run_id": out.name, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out / "run_status.json", status)
    try:
        artifact_info, item = check_preflight(args.slot)
        rows, support = load_masked_rows()
        texts = [row["text"] for row in rows]
        artifact = ROOT / artifact_info["artifact"]
        print(f"Loaded {len(rows)} frozen primary Teke rows for {args.slot}", flush=True)
        if args.slot == "linear_svm":
            predictions = predict_svm(texts, artifact)
        elif args.slot == "bilstm":
            predictions = predict_bilstm(texts, artifact, item, args.device, args.batch_size)
        else:
            predictions = predict_transformer(args.slot, texts, artifact, item, args.device, args.batch_size)
        write_outputs(out, args.slot, rows, predictions, support, artifact_info)
        status["status"] = "complete"
        print(f"Direct transfer complete: {out.relative_to(ROOT)}", flush=True)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run_status.json", status)


if __name__ == "__main__":
    main()
