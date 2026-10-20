"""R09 frozen BiLSTM CPU logits; no fitting, calibration or performance metrics."""
from __future__ import annotations

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

from scripts.extract_professor_teke_svm_scores import CONTRACT_PIN, LABELS, gate, read_json, require
from scripts.run_teke_direct_transfer import load_masked_rows
from scripts.run_stage2_neural_baselines import BiLSTMSentimentClassifier, encode_text
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.preprocessing import canonical_json_sha256

RUN_ID = "professor_round1_teke_scores_v1_bilstm"
SOURCE_RUN = "s2_neural_bilstm_winvoker_seed52_protocol_v2"
SOURCE = ROOT / "outputs/experiments" / SOURCE_RUN
HISTORICAL = ROOT / "outputs/external/teke_direct_transfer_v1_bilstm"
SVM_SCORES = ROOT / "outputs/external/professor_round1_teke_scores_v1_linear_svm"
SVM_SCORE_PIN = "a984c35412238738fbe962d2dffd023ce08c703d12d9e2f537dd3af288f7fe14"
BATCH_SIZE = 32
POOLING = "concatenated_final_bidirectional_lstm_hidden_states"


def tensor_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def extraction_gate():
    before, contract = gate()  # Accepted static pins and every frozen input.
    require(read_json(SVM_SCORES / "run_status.json")["status"] == "complete" and
            sha256_file(SVM_SCORES / "scores.csv") == SVM_SCORE_PIN,
            "Accepted original SVM extraction missing or changed")
    svm_check = read_json(SVM_SCORES / "reproduction_check.json")
    require(svm_check["historical_prediction_disagreements"] == 0 and
            svm_check["score_argmax_disagreements"] == 0 and svm_check["state_unchanged"] and
            read_json(SVM_SCORES / "preservation_check.json")["inputs_unchanged"],
            "Original SVM extraction acceptance gates changed")
    for path in [*SVM_SCORES.iterdir(), Path(__file__).resolve(), ROOT / "src/preprocessing.py"]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return before, contract


def extract(run):
    before, contract = extraction_gate()
    run.write_json("input_inventory.json", before)
    captured = []
    try:
        pin = contract["historical"]["bilstm"]
        artifact = SOURCE / "checkpoints/best_checkpoint.pt"
        require(sha256_file(artifact) == pin["checkpoint_sha256"] and
                sha256_file(HISTORICAL / "predictions.csv") == pin["payloads"]["predictions.csv"],
                "Frozen BiLSTM checkpoint/historical predictions changed")
        rows, support = load_masked_rows()
        with (HISTORICAL / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            historic = list(csv.DictReader(stream))
        require(len(rows) == len(historic) == 67964 and
                support["primary"] == contract["primary_support"] and
                support["short_text_sensitivity"] == contract["short_support"], "Unexpected row support")
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
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            # This local checkpoint is trusted only after matching its frozen SHA.
            checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
            config, vocab = checkpoint["model_config"], checkpoint["vocab"]
            metadata = read_json(SOURCE / "checkpoint_metadata.json")
            vocab_sha = canonical_json_sha256(vocab)
            require(checkpoint["run_identity"]["run_id"] == SOURCE_RUN and
                    vocab_sha == checkpoint["vocab_sha256"] == metadata["vocab_sha256"] and
                    checkpoint["run_identity"]["profile_config_sha256"] == contract["profile_sha256"] and
                    metadata["pooling"] == POOLING, "Checkpoint/vocabulary/profile/pooling mismatch")
            require({key: config[key] for key in
                     ("max_length", "embedding_dim", "hidden_dim", "num_classes", "dropout")} ==
                    {"max_length": 128, "embedding_dim": 300, "hidden_dim": 128,
                     "num_classes": 3, "dropout": 0.3}, "Frozen model configuration mismatch")
            model = BiLSTMSentimentClassifier(
                vocab_size=len(vocab), embedding_dim=int(config["embedding_dim"]),
                hidden_dim=int(config["hidden_dim"]), num_classes=int(config["num_classes"]),
                dropout=float(config["dropout"])).to("cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
                    "BiLSTM must be CPU float32")
            model.eval()
            model.requires_grad_(False)
            state_before = tensor_state_sha256(model)
            run.write_json("checkpoint_record.json", {"source_run": SOURCE_RUN,
                "checkpoint_sha256": pin["checkpoint_sha256"], "model_config": config,
                "vocab_sha256": vocab_sha, "vocab_entries": len(vocab), "pooling": POOLING,
                "encoder": "scripts.run_stage2_neural_baselines.encode_text",
                "class_order": LABELS, "device": "cpu", "precision": "float32",
                "batch_size": BATCH_SIZE, "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "state_before_sha256": state_before, "evaluation_mode": True,
                "training": False, "calibration": False})
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
                        encoded = [encode_text(r["text"], vocab, int(config["max_length"])) for r in batch]
                        ids = torch.tensor([e[0] for e in encoded], dtype=torch.long, device="cpu")
                        attention = torch.tensor([e[1] for e in encoded], dtype=torch.long, device="cpu")
                        scores = model(ids, attention).cpu().numpy()
                        require(scores.shape == (len(batch), 3) and np.isfinite(scores).all(),
                                "Invalid/nonfinite BiLSTM logits")
                        predictions = np.argmax(scores, axis=1)
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
                            print(f"Teke frozen BiLSTM logits: {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
            state_after = tensor_state_sha256(model)
            vocab_after = canonical_json_sha256(vocab)
            run.write_json("reproduction_check.json", {"rows": count,
                "historical_prediction_disagreements": disagreements,
                "gold_polarity_score_ties": ties, "state_unchanged": state_before == state_after,
                "state_after_sha256": state_after, "vocab_unchanged": vocab_sha == vocab_after,
                "score_seconds": time.perf_counter() - score_started})
            require(count == 67964 and disagreements == 0, "Historical labels not reproduced; investigate")
            require(state_before == state_after and vocab_sha == vocab_after, "Frozen model/vocabulary changed")
        run.write_json("results.json", {"status": "scores_extracted_pending_review", "rows": count,
            "source_run": SOURCE_RUN, "checkpoint_sha256": pin["checkpoint_sha256"],
            "historical_predictions_sha256": pin["payloads"]["predictions.csv"],
            "contract_sha256": CONTRACT_PIN, "source_zip_sha256": contract["source_zip_sha256"],
            "profile_sha256": contract["profile_sha256"], "mask_sha256": contract["mask_sha256"],
            "ordered_sample_ids_sha256": ordered, "primary_support": support["primary"],
            "short_support": support["short_text_sensitivity"], "score_kind": "neural_logits",
            "score_order": LABELS, "device": "cpu", "training": False, "calibration": False,
            "performance_metrics": False, "raw_text_exported": False,
            "scientific_acceptance": "pending_returned_artifact_review"})
        run.write_text("report.md", "# Frozen BiLSTM Teke logits\n\nCPU float32 eval/inference_mode "
            "with matching checkpoint/vocabulary/encoder/pooling. All primary rows saved with "
            "three logits and original lineage, no raw text. Zero historical prediction disagreement "
            "and unchanged model/vocabulary/input state required. No fitting, calibration, metrics, "
            "checkpoint reselection or mask change. Inspect returned scores before GPU-slot handoff.\n")
    finally:
        run.write_json("warnings.json", [{"category": w.category.__name__, "message": str(w.message)}
                                         for w in captured])
        after = {name: sha256_file(ROOT / name) for name in before}
        run.write_json("preservation_check.json", {"inputs_unchanged": before == after,
                                                   "checked_files": len(before)})
        require(before == after, "Protected input changed")
    run.declare_complete(["scores.csv", "input_inventory.json", "checkpoint_record.json",
                          "reproduction_check.json", "warnings.json", "preservation_check.json",
                          "results.json", "report.md"])


def main():
    output = ROOT / "outputs/external" / RUN_ID
    require(not output.exists(), "Refusing existing score-extraction directory")
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_frozen_teke_bilstm_score_extraction",
                       config={"contract_sha256": CONTRACT_PIN, "source_run": SOURCE_RUN,
                               "device": "cpu", "precision": "float32", "batch_size": BATCH_SIZE,
                               "max_length": 128, "pooling": POOLING, "training": False,
                               "calibration": False, "performance_metrics": False}) as run:
        extract(run)


if __name__ == "__main__":
    main()
