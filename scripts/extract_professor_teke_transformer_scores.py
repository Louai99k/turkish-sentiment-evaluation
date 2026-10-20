"""Frozen Teke Transformer logits, one CUDA slot; no fit/calibration/metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from scripts.extract_professor_teke_bilstm_scores import extraction_gate, tensor_state_sha256
from scripts.extract_professor_teke_svm_scores import CONTRACT_PIN, LABELS, read_json, require
from scripts.preflight_teke_external_inference import ROSTER
from scripts.run_teke_direct_transfer import load_masked_rows
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.model import BERTurkSentimentClassifier
from src.transformer_model import TransformerSentimentClassifier

BILSTM = ROOT / "outputs/external/professor_round1_teke_scores_v1_bilstm"
BILSTM_SCORE_PIN = "30afd26a87f0242f193091fdf63dbcd63734ffd4966e847aa5484fbbdd1ff84c"
BATCH_SIZE = 32


def gate():
    before, contract = extraction_gate()
    require(read_json(BILSTM / "run_status.json")["status"] == "complete" and
            sha256_file(BILSTM / "scores.csv") == BILSTM_SCORE_PIN, "Accepted BiLSTM scores changed")
    check = read_json(BILSTM / "reproduction_check.json")
    require(check["rows"] == 67964 and check["historical_prediction_disagreements"] == 0 and
            check["state_unchanged"] and check["vocab_unchanged"] and
            read_json(BILSTM / "preservation_check.json")["inputs_unchanged"], "BiLSTM gates changed")
    manifest = read_json(BILSTM / "artifact_manifest.json")
    require(manifest["run_id"] == BILSTM.name and manifest["run_status"] == "complete",
            "BiLSTM manifest identity mismatch")
    for entry in manifest["files"]:
        name = entry["relative_to_run"]
        require(Path(name).name == name, "Unsafe BiLSTM manifest path")
        path = BILSTM / name
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"],
                f"BiLSTM manifest payload changed: {name}")
    for name, expected in read_json(BILSTM / "input_inventory.json").items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha256_file(path) == expected,
                f"BiLSTM recorded input changed: {name}")
    for path in [*BILSTM.iterdir(), Path(__file__).resolve()]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return before, contract


def extract(run, slot):
    before, contract = gate()
    run.write_json("input_inventory.json", before)
    captured = []
    try:
        require(torch.cuda.is_available(), "CUDA unavailable; do not substitute CPU or checkpoint")
        item = next(i for i in ROSTER if i["slot"] == slot)
        source = ROOT / "outputs/experiments" / item["run"]
        historical = ROOT / "outputs/external" / f"teke_direct_transfer_v1_{slot}"
        artifact = source / item["artifact"]
        pin = contract["historical"][slot]
        require(sha256_file(artifact) == pin["checkpoint_sha256"] and
                sha256_file(historical / "predictions.csv") == pin["payloads"]["predictions.csv"],
                "Frozen checkpoint/historical external predictions changed")
        rows, support = load_masked_rows()
        with (historical / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            historic = list(csv.DictReader(stream))
        require(len(rows) == len(historic) == 67964 and
                support["primary"] == contract["primary_support"] and
                support["short_text_sensitivity"] == contract["short_support"], "Primary/support mismatch")
        ordered = hashlib.sha256("\n".join(r["sample_id"] for r in rows).encode("utf-8")).hexdigest()
        require(ordered == contract["ordered_sample_ids_sha256"], "Loaded row order changed")
        for row, old in zip(rows, historic):
            require(row["sample_id"] == old["sample_id"] and
                    int(row["excel_row"]) == int(old["excel_row"]) and
                    row["clean_text_sha256"] == old["clean_text_sha256"] and
                    row["true_label"] == old["true_label"] and
                    LABELS.index(row["true_label"]) == int(old["true_id"]) and
                    row["short_text_sensitivity_included"] ==
                    (old["short_text_sensitivity_included"] == "True"), "Historical row lineage mismatch")
        metadata = read_json(source / "checkpoint_metadata.json")["metadata"]
        require(metadata["experiment_id"] == item["run"] and
                metadata["model_id"] == metadata["tokenizer_id"] == item["model_id"] and
                metadata["model_revision"] == metadata["tokenizer_revision"] == item["model_revision"] and
                metadata["max_length"] == 128 and metadata["precision_mode"] == "float32" and
                metadata["loss_mode"] == "unweighted", "Frozen model/tokenizer/precision mismatch")
        pooling = "pooler_output" if slot == "berturk" else "first_token"
        if slot == "xlm_roberta":
            require(metadata["pooling"] == pooling and
                    metadata["preprocessing"]["profile_config_sha256"] == contract["profile_sha256"],
                    "XLM-R pooling/profile mismatch")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer_id"],
                revision=metadata["tokenizer_revision"], local_files_only=True)
            if slot == "berturk":
                model = BERTurkSentimentClassifier(model_name=metadata["model_id"],
                    model_revision=metadata["model_revision"], num_classes=3,
                    dropout=float(metadata["dropout"]), local_files_only=True)
            else:
                model = TransformerSentimentClassifier(model_id=metadata["model_id"],
                    model_revision=metadata["model_revision"], pooling=pooling, num_classes=3,
                    dropout=float(metadata["dropout"]), local_files_only=True)
            # Trusted pinned local checkpoint only; no optimizer state is restored.
            checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
            for key in ("experiment_id", "model_id", "model_revision", "tokenizer_id",
                        "tokenizer_revision", "max_length", "precision_mode", "loss_mode"):
                require(checkpoint["metadata"][key] == metadata[key], "Serialized checkpoint identity mismatch")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            model = model.to("cuda")
            require(all(p.device.type == "cuda" and p.dtype == torch.float32 for p in model.parameters()),
                    "Frozen Transformer requires CUDA float32")
            model.eval()
            model.requires_grad_(False)
            state_before = tensor_state_sha256(model)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            run.write_json("checkpoint_record.json", {"source_run": item["run"],
                "checkpoint_sha256": pin["checkpoint_sha256"], "metadata": metadata,
                "pooling": pooling, "backbone_access": "model.bert" if slot == "berturk" else "model.backbone",
                "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_size": len(tokenizer),
                "tokenizer_id": metadata["tokenizer_id"], "tokenizer_revision": metadata["tokenizer_revision"],
                "local_files_only": True, "class_order": LABELS, "max_length": 128,
                "padding": "max_length", "add_special_tokens": True, "truncation": True,
                "device": "cuda", "precision": "float32", "autocast": False,
                "batch_size": BATCH_SIZE, "cuda_device": torch.cuda.get_device_name(),
                "cuda_free_after_load_bytes": free_bytes, "cuda_total_bytes": total_bytes,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32, "state_before_sha256": state_before})
            disagreements = ties = count = 0
            score_started = time.perf_counter()
            fields = ["sample_id", "excel_row", "clean_text_sha256", "true_label", "true_id",
                      "short_text_sensitivity_included", "historical_predicted_id", "predicted_id",
                      "predicted_label", "logit_positive", "logit_negative", "logit_notr"]
            with run.path("scores.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                with torch.inference_mode():
                    for start in range(0, len(rows), BATCH_SIZE):
                        batch = rows[start:start + BATCH_SIZE]
                        tokens = tokenizer([r["text"] for r in batch], add_special_tokens=True,
                            max_length=128, padding="max_length", truncation=True,
                            return_attention_mask=True, return_tensors="pt")
                        scores = model(tokens["input_ids"].to("cuda"),
                                       tokens["attention_mask"].to("cuda")).cpu().numpy()
                        require(scores.shape == (len(batch), 3) and np.isfinite(scores).all(),
                                "Invalid/nonfinite Transformer logits")
                        predictions = np.argmax(scores, axis=1)
                        if start == 0:
                            mismatch = sum(int(p) != int(historic[j]["predicted_id"])
                                           for j, p in enumerate(predictions))
                            run.write_json("runtime_gate.json", {"rows": len(batch),
                                "finite_logits": True, "historical_disagreements": mismatch,
                                "cuda_float32_eval_inference_only": True})
                            require(mismatch == 0, "First-batch historical labels differ; stop before full scoring")
                        for j, (row, score, pred) in enumerate(zip(batch, scores, predictions)):
                            old = historic[start + j]
                            disagreements += int(int(pred) != int(old["predicted_id"]))
                            ties += int(row["true_label"] in LABELS[:2] and score[0] == score[1])
                            writer.writerow({"sample_id": row["sample_id"], "excel_row": row["excel_row"],
                                "clean_text_sha256": row["clean_text_sha256"], "true_label": row["true_label"],
                                "true_id": LABELS.index(row["true_label"]),
                                "short_text_sensitivity_included": row["short_text_sensitivity_included"],
                                "historical_predicted_id": old["predicted_id"], "predicted_id": int(pred),
                                "predicted_label": LABELS[int(pred)], "logit_positive": float(score[0]),
                                "logit_negative": float(score[1]), "logit_notr": float(score[2])})
                            count += 1
                        if (start // BATCH_SIZE + 1) % 100 == 0:
                            print(f"Teke frozen {slot} logits: {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
            score_seconds = time.perf_counter() - score_started
            state_after = tensor_state_sha256(model)
            run.write_json("reproduction_check.json", {"rows": count,
                "historical_prediction_disagreements": disagreements, "gold_polarity_score_ties": ties,
                "state_unchanged": state_before == state_after, "state_after_sha256": state_after,
                "score_seconds": score_seconds, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            require(count == 67964 and disagreements == 0, "Historical labels not reproduced; investigate")
            require(state_before == state_after, "Frozen model tensors changed")
        run.write_json("results.json", {"status": "scores_extracted_pending_review", "rows": count,
            "model_slot": slot, "source_run": item["run"], "checkpoint_sha256": pin["checkpoint_sha256"],
            "historical_predictions_sha256": pin["payloads"]["predictions.csv"],
            "contract_sha256": CONTRACT_PIN, "source_zip_sha256": contract["source_zip_sha256"],
            "mask_sha256": contract["mask_sha256"], "profile_sha256": contract["profile_sha256"],
            "ordered_sample_ids_sha256": ordered, "primary_support": support["primary"],
            "short_support": support["short_text_sensitivity"], "score_kind": "neural_logits",
            "score_order": LABELS, "training": False, "calibration": False, "performance_metrics": False,
            "raw_text_exported": False, "scientific_acceptance": "pending_returned_artifact_review"})
        run.write_text("report.md", "# Frozen Transformer Teke logits\n\nCUDA float32 eval/inference_mode; "
            "offline pinned model/tokenizer/checkpoint and original profile/masks/IDs. First-batch "
            "runtime check and full historical-label reproduction required. All primary rows retain "
            "three logits, no raw text. No training/calibration/reselection/performance metrics. "
            "Inspect returned scores/warnings/state/preservation before further slots/metrics.\n")
    finally:
        run.write_json("warnings.json", [{"category": w.category.__name__, "message": str(w.message)}
                                         for w in captured])
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after,
                                                   "checked_files": len(before)})
        require(before == after, "Protected input changed")
    run.declare_complete(["scores.csv", "runtime_gate.json", "input_inventory.json", "checkpoint_record.json",
                          "reproduction_check.json", "warnings.json", "preservation_check.json",
                          "results.json", "report.md"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=("berturk", "xlm_roberta"), required=True)
    args = parser.parse_args()
    run_id = f"professor_round1_teke_scores_v1_{args.slot}"
    output = ROOT / "outputs/external" / run_id
    require(not output.exists(), "Refusing existing score-extraction directory")
    with ExperimentRun(output, run_id=run_id, run_kind="cuda_frozen_teke_transformer_score_extraction",
                       config={"model_slot": args.slot, "contract_sha256": CONTRACT_PIN,
                               "device": "cuda", "precision": "float32", "batch_size": BATCH_SIZE,
                               "max_length": 128, "training": False, "calibration": False,
                               "performance_metrics": False}) as run:
        extract(run, args.slot)


if __name__ == "__main__":
    main()
