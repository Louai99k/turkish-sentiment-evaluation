"""Validate frozen extraction inputs, or extract representations after the gate.

The validate stage does not import torch/transformers or deserialize checkpoints.
It checks source syntax, declared scope, hashes and sample lineage. Runtime
compatibility is established by extract's saved-logit and label checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_constants import WINVOKER_TEST_MASK_PATH, WINVOKER_TEST_MASK_SHA256

CONFIG = ROOT / "configs/analysis/berturk_extraction_v1.json"
GATE = ROOT / "outputs/audits/berturk_extraction_static_v1"
FILES = ("representation_sample.csv", "attention_sample.csv", "error_coding_sample.csv")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(directory):
    manifest = read(directory / "artifact_manifest.json")
    require(manifest["run_status"] == "complete", f"Incomplete manifest: {directory}")
    require(read(directory / "run_status.json")["status"] == "complete", f"Incomplete run: {directory}")
    for entry in manifest["files"]:
        path = (directory / entry["relative_to_run"]).resolve()
        require(path.is_relative_to(directory.resolve()), "Manifest path escapes directory")
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"], f"Hash/size mismatch: {path}")


def inputs(config):
    for key, expected in {"max_length": 128, "dropout": 0.3, "pooling": "pooler_output",
                          "precision": "float32", "attention_backend": "eager"}.items():
        require(config[key] == expected, f"Unsupported frozen extraction setting: {key}")
    require(isinstance(config["batch_size"], int) and config["batch_size"] > 0, "Invalid batch size")
    plan_dir = ROOT / config["plan_directory"]
    require(sha256_file(plan_dir / "artifact_manifest.json") == config["plan_manifest_sha256"], "Frozen plan manifest changed")
    verify_manifest(plan_dir)
    plan = read(plan_dir / "frozen_plan.json")
    selected = read(plan_dir / "selected_checkpoint.json")
    require(selected == plan["selected_checkpoint"], "Plan checkpoint mismatch")
    require(selected["experiment_id"] == config["experiment_id"], "Wrong experiment")
    require(selected["model_revision"] == selected["tokenizer_revision"] == config["revision"], "Revision mismatch")
    source = ROOT / "outputs/experiments" / selected["experiment_id"]
    verify_manifest(source)
    source_cfg = read(source / "config.json")
    metadata = read(source / "checkpoint_metadata.json")
    for key, expected in {"model_id": config["model_id"], "tokenizer_id": config["model_id"],
                          "model_revision": config["revision"], "tokenizer_revision": config["revision"],
                          "max_length": 128, "dropout": 0.3, "loss_mode": "unweighted", "seed": 52}.items():
        require(source_cfg[key] == expected, f"Source config mismatch: {key}")
    require(metadata["metadata"]["experiment_id"] == selected["experiment_id"], "Checkpoint identity mismatch")
    require(metadata["validation_weighted_f1"] == selected["validation_weighted_f1"], "Selection score mismatch")
    for key in ("checkpoint", "predictions"):
        path = ROOT / selected[f"{key}_path"]
        require(path.resolve().is_relative_to(source.resolve()), "Source path escapes selected run")
        require(sha256_file(path) == selected[f"{key}_sha256"], f"Selected {key} hash mismatch")
    require(sha256_file(WINVOKER_TEST_MASK_PATH) == WINVOKER_TEST_MASK_SHA256, "Authoritative mask changed")
    lineage = pd.read_csv(WINVOKER_TEST_MASK_PATH, keep_default_na=False)
    predictions = pd.read_csv(ROOT / selected["predictions_path"], keep_default_na=False)
    require(len(predictions) == len(lineage) == 88136, "Incorrect population size")
    require(predictions.sample_id.is_unique, "Duplicate source sample IDs")
    require(predictions.test_position.tolist() == list(range(88136)), "Prediction ordering changed")
    require(lineage.test_position.tolist() == list(range(88136)), "Lineage ordering changed")
    for column, ref in (("true_id", "label_id"), ("source", "dataset"),
                        ("source_row_index", "source_row_index"), ("strict_included", "strict_included"),
                        ("strict_unique_included", "strict_unique_included")):
        require(predictions[column].tolist() == lineage[ref].tolist(), f"Population lineage mismatch: {column}")
    text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in predictions.text]
    require(text_hashes == lineage.cleaned_text_sha256.tolist() == predictions.cleaned_text_sha256.tolist(), "Text lineage mismatch")
    ids = [f"winvoker-train-4bd9103ad5f2-{int(i):06d}" for i in lineage.source_row_index]
    require(predictions.sample_id.tolist() == ids, "Stable-ID mismatch")
    indexed = predictions.set_index("sample_id", drop=False)
    samples = {}
    for filename, count in zip(FILES, (3000, 3, 60)):
        sample = pd.read_csv(plan_dir / filename, keep_default_na=False)
        require(len(sample) == count and sample.sample_id.is_unique, f"Invalid sample: {filename}")
        rows = indexed.loc[sample.sample_id].reset_index(drop=True)
        for column in set(sample.columns) & set(rows.columns):
            require(sample[column].tolist() == rows[column].tolist(), f"Sample content mismatch: {filename}/{column}")
        require(rows.strict_included.eq(True).all(), "Sample includes non-strict row")
        if filename.startswith("representation"):
            require(rows.strict_unique_included.eq(True).all(), "Non-unique representation row")
            require(rows.true_id.value_counts().to_dict() == {0: 1000, 1: 1000, 2: 1000}, "Unbalanced sample")
        samples[filename] = rows
    identity = {
        "config_sha256": sha256_file(CONFIG), "runner_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(ROOT / "src/model.py"),
        "artifact_writer_sha256": sha256_file(ROOT / "src/experiment_artifacts.py"),
        "plan_manifest_sha256": config["plan_manifest_sha256"],
        "source_manifest_sha256": sha256_file(source / "artifact_manifest.json"),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "mask_sha256": WINVOKER_TEST_MASK_SHA256,
        "sample_hashes": {f: sha256_file(plan_dir / f) for f in FILES},
    }
    return selected, samples, identity, metadata


def validate(run, config):
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            require(node.func.attr not in {"backward", "step", "fit", "fit_transform"}, "Unexpected fitting/training call")
    selected, samples, identity, _ = inputs(config)
    run.write_json("gate.json", {"status": "passed", "identity": identity,
                                "selected_checkpoint": selected,
                                "sample_counts": {k: len(v) for k, v in samples.items()},
                                "model_loaded": False, "checkpoint_deserialized": False,
                                "runtime_compatibility": "pending saved-logit/label reproduction at extraction"})
    run.write_text("report.md", "# BERTurk extraction preflight\n\nSyntax, scope inspection, frozen hashes, and all sample identities passed.\n\n"
                   "No model, tokenizer or checkpoint tensors were loaded. Runtime compatibility is still pending. "
                   "Extraction retains the selected unweighted seed-52 configuration and original sample order.\n")
    run.declare_complete(["gate.json", "report.md"])


def extract(run, config, device):
    selected, samples, identity, metadata = inputs(config)
    verify_manifest(GATE)
    gate = read(GATE / "gate.json")
    require(gate["status"] == "passed" and gate["identity"] == identity, "Preflight is missing or stale")
    # All expensive imports and checkpoint deserialization occur after the gate.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    from src.model import BERTurkSentimentClassifier

    require(device != "cuda" or torch.cuda.is_available(), "Requested CUDA is unavailable")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True)
    model = BERTurkSentimentClassifier(model_name=config["model_id"], model_revision=config["revision"],
                                       dropout=config["dropout"], local_files_only=True)
    checkpoint = torch.load(ROOT / selected["checkpoint_path"], map_location="cpu", weights_only=True)
    require(checkpoint["metadata"] == metadata["metadata"], "Deserialized checkpoint metadata mismatch")
    require(checkpoint["epoch"] == metadata["epoch_zero_based"] and checkpoint["val_f1"] == selected["validation_weighted_f1"], "Checkpoint selection mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    require(model.bert.config._commit_hash == config["revision"], "Loaded model revision mismatch")
    require(callable(getattr(model.bert, "set_attn_implementation", None)),
            "Installed backbone lacks explicit attention-backend switching; stop for compatibility review")
    model.bert.set_attn_implementation(config["attention_backend"])
    require(model.bert.config._attn_implementation == "eager", "Attention backend was not applied")
    model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    frames = [samples[f] for f in FILES]
    union = pd.concat(frames, ignore_index=True).drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    vectors, logits, token_counts = [], [], []
    attention_rows = []
    attention_ids = set(samples["attention_sample.csv"].sample_id)
    with torch.inference_mode():
        for start in range(0, len(union), config["batch_size"]):
            batch = union.iloc[start:start+config["batch_size"]]
            encoded = tokenizer(batch.text.tolist(), add_special_tokens=True, padding="max_length", max_length=config["max_length"],
                                truncation=True, return_attention_mask=True, return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            need_attention = any(s in attention_ids for s in batch.sample_id)
            output = model.bert(input_ids=ids, attention_mask=mask, output_attentions=need_attention)
            pooled = output.pooler_output
            scores = model.classifier(model.dropout(pooled))
            actual = scores.cpu().numpy()
            expected = batch[["logit_positive", "logit_negative", "logit_notr"]].to_numpy(dtype=np.float32)
            require(np.isfinite(actual).all() and torch.isfinite(pooled).all().item(), "Non-finite extraction")
            require(np.allclose(actual, expected, atol=config["logit_absolute_tolerance"], rtol=config["logit_relative_tolerance"]),
                    f"Saved-logit mismatch at batch {start}; max absolute delta {float(np.max(np.abs(actual-expected)))}")
            require(np.array_equal(actual.argmax(1), batch.predicted_id.to_numpy()), "Saved-label mismatch")
            vectors.extend(pooled.cpu().numpy()); logits.extend(actual)
            lengths = mask.sum(1).cpu().tolist()
            token_counts.extend(lengths)
            if need_attention:
                require(output.attentions is not None, "Backend did not return attention")
                for i, sample_id in enumerate(batch.sample_id):
                    if sample_id not in attention_ids:
                        continue
                    attention = output.attentions[-1][i].mean(dim=0)[0].cpu().numpy()
                    token_ids = ids[i, :lengths[i]].cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    require(np.isfinite(attention).all() and np.all(attention >= 0), "Invalid attention weights")
                    require(abs(float(attention[:lengths[i]].sum())-1) < 1e-5, "Attention mass check failed")
                    for pos, (token_id, token) in enumerate(zip(token_ids, tokens)):
                        attention_rows.append({"sample_id": sample_id, "token_position": pos, "token_id": token_id,
                                               "token": token, "special_token": token_id in tokenizer.all_special_ids,
                                               "mean_head_cls_attention": float(attention[pos])})
            print(f"Extracted and reproduced {min(start+len(batch), len(union))}/{len(union)} samples", flush=True)
    matrix = np.asarray(vectors, dtype=np.float32)
    # The first frame is exactly the frozen representation order.
    require(union.sample_id.iloc[:3000].tolist() == samples["representation_sample.csv"].sample_id.tolist(), "Representation order changed")
    np.save(run.path("berturk_pooler_output.npy"), matrix[:3000], allow_pickle=False)
    np.save(run.path("extracted_logits.npy"), np.asarray(logits, dtype=np.float32), allow_pickle=False)
    union["retained_token_count"] = token_counts
    union["untruncated_token_count"] = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in union.text]
    union["was_truncated"] = union.untruncated_token_count > config["max_length"]
    run.write_csv("sample_index.csv", union[["sample_id", "test_position", "source", "true_id", "predicted_id", "retained_token_count", "untruncated_token_count", "was_truncated"]])
    run.write_csv("attention_tokens.csv", attention_rows)
    run.write_json("extraction_summary.json", {"identity": identity, "selected_checkpoint": selected,
        "representation_shape": list(matrix[:3000].shape), "samples_reproduced": len(union),
        "maximum_absolute_logit_difference": float(np.max(np.abs(np.asarray(logits)-union[["logit_positive", "logit_negative", "logit_notr"]].to_numpy()))),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "attention_backend": model.bert.config._attn_implementation,
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_revision": config["revision"],
        "tokenizer_vocabulary_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        "pooling": "pooler_output", "training": False, "all_labels_reproduced": True})
    run.write_text("report.md", "# BERTurk extraction\n\nFrozen unweighted seed-52 samples reproduced saved labels and logits within declared tolerances. "
                   "The first 3,000 sample-index rows identify pooler-output matrix rows. Attention is final-layer mean-head CLS inspection, not causal explanation. "
                   "Truncation indicators are recorded; category and irony annotations still require review. "
                   "TF-IDF, projection and figure generation are separate follow-up stages.\n")
    run.declare_complete(["berturk_pooler_output.npy", "extracted_logits.npy", "sample_index.csv", "attention_tokens.csv", "extraction_summary.json", "report.md"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validate", "extract"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = read(CONFIG)
    require(config["schema_version"] == "berturk-extraction-v1", "Unknown schema")
    output = GATE if args.stage == "validate" else ROOT / "outputs/analysis/berturk_qualitative_extraction_v1"
    require(not output.exists(), "Output already exists; preserve earlier evidence")
    with ExperimentRun(output, run_id=output.name, run_kind=f"berturk_qualitative_{args.stage}",
                       config={"stage": args.stage, "extraction_config": config, "config_sha256": sha256_file(CONFIG)}) as run:
        if args.stage == "validate":
            validate(run, config)
        else:
            extract(run, config, args.device)


if __name__ == "__main__":
    main()
