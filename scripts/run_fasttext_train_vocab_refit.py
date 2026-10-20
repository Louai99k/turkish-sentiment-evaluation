"""User-run fastText singleton CPU refit with immutable TRAIN-ONLY vector subset.

--preflight hashes local archive, verifies TRAIN lineage and freezes vocabulary;
no vectors/classifier fitted and no held-out texts tokenized. --fit checks that
gate, streams vectors, fits original-setup LR, then materializes/evaluates holdout.
No download, GPU, tuning, seed sweep, overwrite, or implicit resume.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import struct
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_winvoker_dataset, preprocess_dataset_with_profile
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.preprocessing import (canonical_json_sha256, fingerprint_preprocessed_splits,
                               load_preprocessing_profile, load_profile_fingerprint_manifest,
                               preprocessing_profile_record, verify_split_fingerprints)
from src.protocol_metrics import (build_evaluation_metrics, classification_metrics,
                                  confusion_matrix_rows, flatten_metric_views)
from src.protocol_predictions import (build_winvoker_prediction_frame, load_winvoker_test_lineage,
                                      sha256_text, stable_winvoker_sample_id)


def local(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT.resolve())
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def freeze_train_vocabulary(texts):
    terms = set()
    for text in texts:
        for token in str(text).split():
            terms.add(token)
            terms.add(token.lower())
    return frozenset(terms)


def load_vectors(archive, vocabulary, dimension, declared_tokens):
    vectors = {}
    rows = 0
    with gzip.open(archive, "rt", encoding="utf-8", errors="ignore") as stream:
        if stream.readline().strip().split() != [str(declared_tokens), str(dimension)]:
            raise ValueError("Unexpected fastText archive header")
        # Continue to EOF, even after all requested words found, to check gzip CRC.
        for line in stream:
            rows += 1
            parts = line.rstrip().split(" ")  # Historical parsing preserved.
            if len(parts) >= dimension + 1 and parts[0] in vocabulary:
                values = np.asarray(parts[1:], dtype=np.float32)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid vector for requested token {parts[0]!r}")
                vectors[parts[0]] = values
            if rows % 250000 == 0:
                print(f"Archive rows scanned: {rows}; training-vocabulary vectors loaded: {len(vectors)}", flush=True)
    if rows != declared_tokens:
        raise ValueError(f"Archive row count differs from header: {rows}")
    if not vectors or not set(vectors).issubset(vocabulary):
        raise ValueError("Empty/invalid training-only vector subset")
    digest = hashlib.sha256(b"fasttext-subset-float32-v1\0")
    for token in sorted(vectors):
        encoded = token.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(vectors[token].astype("<f4", copy=False).tobytes())
    return vectors, {"archive_rows_scanned": rows, "full_gzip_crc_checked": True,
                     "loaded_vectors": len(vectors), "vector_dim": dimension,
                     "vector_subset_sha256": digest.hexdigest(), "selection_split": "train_only"}


def vectorize(texts, vectors, dimension):
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    total = matched = exact = fallback = no_matches = zero_rows = 0
    for position, text in enumerate(texts):
        rows = []
        for token in str(text).split():
            total += 1
            value = vectors.get(token)
            if value is not None:
                exact += 1
            else:
                value = vectors.get(token.lower())
                if value is not None:
                    fallback += 1
            if value is not None:
                rows.append(value)
                matched += 1
        if rows:
            matrix[position] = np.mean(rows, axis=0).astype(np.float32)
        else:
            no_matches += 1
        zero_rows += int(not np.any(matrix[position]))
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite sentence embeddings")
    return matrix, {"rows": len(texts), "token_occurrences": total,
                    "matched_occurrences": matched, "exact_matches": exact, "lowercase_fallback_matches": fallback,
                    "oov_occurrences": total - matched, "oov_rate": (total - matched) / total if total else None,
                    "rows_without_matched_tokens": no_matches, "zero_vector_rows": zero_rows,
                    "zero_vector_rate": zero_rows / len(texts), "matrix_bytes": int(matrix.nbytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    config_path = local(args.config)
    config = read(config_path)
    if config.get("schema_version") != "fasttext-train-vocabulary-refit-v1" or (
        config["run_id"], config["preflight_run_id"]) != (
            "rev1_fasttext_logreg_train_vocab_v1", "professor_round1_fasttext_train_vocab_preflight_v1"):
        raise ValueError("Unexpected configuration/run contract")
    if config["feature_protocol"]["vocabulary_split"] != "train_only" or config["selection"] != "singleton_original_setup_no_tuning":
        raise ValueError("Unapproved vocabulary/selection policy")
    run_id = config["preflight_run_id"] if args.preflight else config["run_id"]
    output = ROOT / "outputs" / ("audits" if args.preflight else "experiments") / run_id
    resolved = dict(config, mode="preflight_only" if args.preflight else "cpu_refit",
                    executable_configuration_sha256=sha256_file(config_path))
    with ExperimentRun(output, run_id=run_id, run_kind=resolved["mode"], config=resolved,
                       runtime_environment={"device": "cpu", "network": False}) as run:
        inputs = []
        timings = {}

        def identity(name, expected=None):
            path = local(name)
            observed = sha256_file(path)
            if expected is not None and observed != expected:
                raise ValueError(f"Input SHA-256 mismatch: {name}")
            inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": observed,
                           "bytes": path.stat().st_size})
            return path

        for package, version in config["required_packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Package changed since accepted inventory: {package}")
        inventory = read(identity(config["parameter_inventory"], config["parameter_inventory_sha256"]))
        if config["classifier"] != inventory["models"]["fasttext_logreg"]["estimators"]["classifier"]["parameters"]:
            raise ValueError("Historical LR parameters differ")
        data_config = read(identity(config["dataset_config"], config["dataset_config_sha256"]))
        profile_path = identity(config["preprocessing_profile"], config["preprocessing_profile_sha256"])
        _, profile = load_preprocessing_profile(profile_path)
        profile_record = preprocessing_profile_record(profile_path, profile)
        manifest_path, _, expected = load_profile_fingerprint_manifest(config["fingerprint_manifest"], profile_record)
        identity(manifest_path.relative_to(ROOT))
        identity(config_path.relative_to(ROOT))
        for source in ("scripts/run_fasttext_train_vocab_refit.py", "scripts/run_stage2_neural_baselines.py",
                       "src/data_loader.py", "src/preprocessing.py", "src/experiment_artifacts.py",
                       "src/protocol_predictions.py", "src/protocol_metrics.py", "src/protocol_constants.py"):
            identity(source)
        identity(data_config["historic_split"]["assignments_path"], data_config["historic_split"]["assignments_sha256"])
        identity(data_config["strict_test"]["mask_path"], data_config["strict_test"]["mask_sha256"])
        historical = "outputs/experiments/" + config["historical_run_id"]
        identity(historical + "/model_bundle.pkl", config["historical_bundle_sha256"])
        old_predictions_path = identity(historical + "/predictions.csv")
        identity(historical + "/config.json")
        identity(historical + "/metrics.json")
        print("Checking full local fastText archive SHA-256 (no download)...", flush=True)
        clock = time.perf_counter()
        archive = identity(config["archive"], config["archive_sha256"])
        if archive.stat().st_size != config["archive_size_bytes"]:
            raise ValueError("Archive size mismatch")
        with gzip.open(archive, "rt", encoding="utf-8", errors="ignore") as stream:
            if stream.readline().strip().split() != [str(config["archive_declared_tokens"]), str(config["vector_dim"])]:
                raise ValueError("Archive header mismatch")
        timings["archive_identity_seconds"] = time.perf_counter() - clock
        gate = ROOT / "outputs/audits" / config["preflight_run_id"]
        if args.fit:
            if read(gate / "run_status.json")["status"] != "complete" or read(gate / "input_inventory.json") != inputs:
                raise ValueError("Complete unchanged preflight required")
            for item in read(gate / "artifact_manifest.json")["files"]:
                if sha256_file(gate / item["relative_to_run"]) != item["sha256"]:
                    raise ValueError("Preflight artifact changed")
        run.write_json("input_inventory.json", inputs)
        clock = time.perf_counter()
        print("Loading cached pinned dataset; materializing TRAIN ONLY...", flush=True)
        dataset = load_winvoker_dataset(data_config["dataset_id"], data_config["dataset_revision"])
        train_splits = preprocess_dataset_with_profile(dataset, profile, included_splits=("train",),
                                                       assignments_path=local(data_config["historic_split"]["assignments_path"]))
        actual = fingerprint_preprocessed_splits(train_splits)
        verify_split_fingerprints(actual, expected["splits"])
        train = train_splits["train"]
        needed = freeze_train_vocabulary(train["text"])
        vocabulary_record = {"selection_split": "train_only", "tokenizer": config["feature_protocol"]["tokenizer"],
                             "expansion": config["feature_protocol"]["vocabulary_expansion"],
                             "terms": sorted(needed), "term_count": len(needed)}
        vocabulary_record["vocabulary_sha256"] = canonical_json_sha256({"terms": vocabulary_record["terms"]})
        if args.fit and read(gate / "training_vocabulary.json") != vocabulary_record:
            raise ValueError("Training vocabulary changed after gate")
        run.write_json("training_vocabulary.json", vocabulary_record)
        run.write_json("preprocessing.json", dict(profile_record, verified_split_fingerprints=actual,
                                                 heldout_fingerprints_from_accepted_manifest=expected["splits"]))
        timings["train_data_and_vocabulary_seconds"] = time.perf_counter() - clock
        resources = {"observed_process_rss_bytes_after_train_vocabulary": rss(),
                     "requested_training_terms": len(needed), "train_dense_float32_bytes": len(train) * 300 * 4,
                     "vector_payload_upper_bound_bytes": len(needed) * 300 * 4,
                     "fit_allowance_minutes": [5, 30], "host_ram_allowance_gb": [4, 8],
                     "additional_disk_allowance_gb": [1, 2], "gpu_required": False,
                     "estimate_is_not_benchmark": True, "python_dict_and_solver_copies_not_in_payload_bound": True}
        run.write_json("resources.json", resources)
        if args.preflight:
            run.write_json("results.json", {"input_and_train_vocabulary_gate": "passed", "classifier_fitted": False,
                                            "heldout_texts_materialized": False, "archive_sha256_verified": True,
                                            "full_gzip_crc_checked": False, "fit_ready_requires_receipt_review": True})
            run.write_json("timings.json", timings)
            run.write_text("report.md", "# Training-only fastText preflight\n\nFull compressed archive hash and header "
                           "checked. Training ordered fingerprint/vocabulary frozen; no holdout text materialized, "
                           "no vector scan/classifier fit/inference. Local hash is not publisher-origin proof. "
                           "Review resources before singleton CPU fit. Full CRC checked during eventual EOF scan.\n")
            run.declare_complete(["input_inventory.json", "training_vocabulary.json", "preprocessing.json",
                                  "resources.json", "results.json", "timings.json", "report.md"])
            return
        print("Frozen TRAIN vocabulary; streaming archive to select only these terms...", flush=True)
        clock = time.perf_counter()
        vectors, vector_record = load_vectors(archive, needed, config["vector_dim"], config["archive_declared_tokens"])
        timings["vector_subset_scan_seconds"] = time.perf_counter() - clock
        run.write_json("vector_subset.json", vector_record)
        frozen_vector_keys = frozenset(vectors)
        clock = time.perf_counter()
        x_train, train_coverage = vectorize(train["text"], vectors, config["vector_dim"])
        timings["train_vectorization_seconds"] = time.perf_counter() - clock
        coverage = {"train": train_coverage}
        run.write_json("coverage.json", coverage)
        classifier = LogisticRegression(**config["classifier"])
        clock = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(x_train, train["label"])
        timings["classifier_fit_seconds"] = time.perf_counter() - clock
        del x_train
        run.write_json("timings.json", timings)
        warning_rows = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
        run.write_json("training_record.json", {"warnings": warning_rows,
                       "convergence_warning": any(w["category"] == "ConvergenceWarning" for w in warning_rows),
                       "classifier_n_iter": classifier.n_iter_.tolist(), "classes": classifier.classes_.tolist(),
                       "observed_process_rss_bytes": rss(), "selection": config["selection"],
                       "scientific_unit": "singleton deterministic configuration; not training-seed variability"})
        if not np.array_equal(classifier.classes_, [0, 1, 2]):
            raise ValueError("Noncanonical class ordering")
        # Self-contained research-only bundle: frozen vector subset retained, no absolute cache dependency.
        with run.path("model_bundle.pkl").open("wb") as stream:
            pickle.dump({"classifier": classifier, "vectors": vectors, "vector_dim": config["vector_dim"],
                         "feature_protocol": config["feature_protocol"], "vector_subset": vector_record,
                         "preprocessing": profile_record, "run_id": run_id}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        del vocabulary_record
        # Holdout first materialized AFTER vocabulary/subset freeze and classifier fit.
        heldout = preprocess_dataset_with_profile(dataset, profile, included_splits=("val", "test"),
                                                 assignments_path=local(data_config["historic_split"]["assignments_path"]))
        del dataset
        actual.update(fingerprint_preprocessed_splits(heldout))
        verify_split_fingerprints(actual, expected["splits"])
        run.write_json("preprocessing.json", dict(profile_record, verified_split_fingerprints=actual))
        for name in ("val", "test"):
            frame = heldout[name]
            clock = time.perf_counter()
            x, split_coverage = vectorize(frame["text"], vectors, config["vector_dim"])
            coverage["validation" if name == "val" else "test"] = split_coverage
            predicted = classifier.predict(x)
            scores = classifier.decision_function(x)
            probabilities = classifier.predict_proba(x)
            del x
            timings[name + "_evaluation_seconds"] = time.perf_counter() - clock
            if name == "val":
                predictions = pd.DataFrame({"sample_id": [stable_winvoker_sample_id(int(v)) for v in frame["source_row_index"]],
                    "source_row_index": frame["source_row_index"], "order_in_split": frame["order_in_split"],
                    "source": frame["dataset"], "text": frame["text"], "cleaned_text_sha256": frame["text"].map(sha256_text),
                    "true_id": frame["label"], "predicted_id": predicted})
                for i, label in enumerate(("positive", "negative", "notr")):
                    predictions["decision_score_" + label] = scores[:, i]
            else:
                predictions = build_winvoker_prediction_frame(load_winvoker_test_lineage(), frame["text"],
                                                               frame["label"], predicted, scores, "decision_score")
            for i, label in enumerate(("positive", "negative", "notr")):
                predictions["probability_" + label] = probabilities[:, i]
            run.write_csv("validation_predictions.csv" if name == "val" else "predictions.csv", predictions)
            if name == "val":
                run.write_json("validation_metrics.json", classification_metrics(frame["label"], predicted, subset_id="historic_validation"))
            else:
                metrics = build_evaluation_metrics(predictions)
                run.write_json("metrics.json", metrics)
                run.write_csv("metric_views.csv", flatten_metric_views(metrics))
                run.write_csv("confusion_matrices.csv", confusion_matrix_rows(metrics))
                old = pd.read_csv(old_predictions_path, keep_default_na=False)
                if len(old) != len(frame) or not np.array_equal(old["true_id"], frame["label"]) or not np.array_equal(old["text"], frame["text"]):
                    raise ValueError("Historical comparison row identity mismatch")
                run.write_json("historical_comparison.json", {"historical_run_id": config["historical_run_id"],
                    "changed_test_predictions": int(np.sum(old["predicted_id"].to_numpy() != predicted)),
                    "old_complete": classification_metrics(old["true_id"], old["predicted_id"], subset_id="historic_test_complete"),
                    "new_complete": metrics["views"]["historic_test_complete"],
                    "historical_validation_predictions_available": False,
                    "interpretation": "Authorized train-only-vocabulary correction; no seed sweep or historical-origin repair."})
        if frozenset(vectors) != frozen_vector_keys or not frozen_vector_keys.issubset(needed):
            raise ValueError("Vector subset mutated during holdout evaluation")
        for item in inputs:
            if sha256_file(local(item["path"])) != item["sha256"]:
                raise ValueError(f"Input changed during fit: {item['path']}")
        run.write_json("coverage.json", coverage)
        run.write_json("timings.json", timings)
        run.write_text("report.md", "# Training-only fastText singleton candidate\n\nInspect convergence, coverage, "
                       "lineage, metrics and historical comparison before acceptance/promotion. Original mean embeddings and "
                       "LR settings retained; only train-only vector-vocabulary selection changed. Holdout materialized "
                       "after fit. Frozen subset/self-contained bundle saved research-only. Scores are LR decision scores "
                       "and probabilities, no claimed calibration. Local archive hash/CRC does not establish publisher origin. "
                       "Original artifacts/statistics/external checkpoints/masks unchanged; no automatic manuscript update.\n")
        run.declare_complete(["input_inventory.json", "training_vocabulary.json", "preprocessing.json", "resources.json",
                              "vector_subset.json", "coverage.json", "training_record.json", "model_bundle.pkl",
                              "validation_predictions.csv", "validation_metrics.json", "predictions.csv", "metrics.json",
                              "metric_views.csv", "confusion_matrices.csv", "historical_comparison.json", "timings.json", "report.md"])


if __name__ == "__main__":
    main()
