"""User-run CPU SVM preflight/refit, isolated from accepted historical artifacts.

--preflight performs only input/environment/data-lineage checks, never fits a
vectorizer or classifier. --fit is a separate user-executed one-configuration
refit after accepted preflight. No overwrite/resume, GPU use, tuning or download.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set before importing Hugging Face dependencies. Use only already cached data.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.data_loader import load_winvoker_dataset, preprocess_dataset_with_profile
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.preprocessing import (canonical_json_sha256, fingerprint_preprocessed_splits,
                               load_preprocessing_profile, load_profile_fingerprint_manifest,
                               preprocessing_profile_record, verify_split_fingerprints)
from src.protocol_metrics import (build_evaluation_metrics, classification_metrics,
                                  confusion_matrix_rows, flatten_metric_views)
from src.protocol_predictions import (build_winvoker_prediction_frame,
                                      load_winvoker_test_lineage, sha256_text,
                                      stable_winvoker_sample_id)


def local(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT.resolve())
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def process_rss_bytes():
    # Linux /proc measures process RSS, unlike host/container physical RAM.
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    config_path = local(args.config)
    config = read(config_path)
    if config.get("schema_version") != "svm-complete-record-refit-v1":
        raise ValueError("Unsupported schema")
    allowed_runs = {
        ("rev1_tfidf_svm_winvoker_complete_records_v1", "professor_round1_svm_refit_preflight_v1"),
        ("rev1_tfidf_svm_winvoker_complete_records_v2", "professor_round1_svm_refit_preflight_v2"),
    }
    if (config["run_id"], config["preflight_run_id"]) not in allowed_runs:
        raise ValueError("Unexpected run contract")
    if config["fit_split"] != "train_only" or config["selection"] != "one_original_setup_fit_no_tuning":
        raise ValueError("Unapproved fitting/selection contract")
    run_id = config["preflight_run_id"] if args.preflight else config["run_id"]
    output = ROOT / "outputs" / ("audits" if args.preflight else "experiments") / run_id
    resolved = dict(config, mode="preflight_only" if args.preflight else "cpu_refit",
                    executable_configuration_sha256=sha256_file(config_path))
    with ExperimentRun(output, run_id=run_id, run_kind=resolved["mode"], config=resolved,
                       runtime_environment={"device": "cpu", "network": False}) as run:
        inputs = []

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
                raise ValueError(f"Package changed since inventory: {package}; reconcile before fitting")
        inventory = read(identity(config["parameter_inventory"], config["parameter_inventory_sha256"]))
        for key, record in (("vectorizer", "vectorizer"), ("classifier", "model")):
            if config[key] != inventory["models"]["svm"]["estimators"][record]["parameters"]:
                raise ValueError(f"Historical parameter mismatch: {key}")
        data_config = read(identity(config["dataset_config"], config["dataset_config_sha256"]))
        profile_path = identity(config["preprocessing_profile"], config["preprocessing_profile_sha256"])
        _, profile = load_preprocessing_profile(profile_path)
        profile_record = preprocessing_profile_record(profile_path, profile)
        fingerprint_path, _, expected = load_profile_fingerprint_manifest(config["fingerprint_manifest"], profile_record)
        identity(fingerprint_path.relative_to(ROOT))
        identity(config_path.relative_to(ROOT))
        for source in ("scripts/run_svm_complete_records.py", "src/data_loader.py", "src/preprocessing.py",
                       "src/experiment_artifacts.py", "src/protocol_predictions.py", "src/protocol_metrics.py",
                       "src/protocol_constants.py"):
            identity(source)
        identity(data_config["historic_split"]["assignments_path"], data_config["historic_split"]["assignments_sha256"])
        identity(data_config["strict_test"]["mask_path"], data_config["strict_test"]["mask_sha256"])
        historical_folder = "outputs/experiments/" + config["historical_run_id"]
        historical_bundle = identity(historical_folder + "/model_bundle.pkl", config["historical_bundle_sha256"])
        old_predictions_path = identity(historical_folder + "/predictions.csv")
        identity(historical_folder + "/config.json")
        identity(historical_folder + "/metrics.json")
        if args.fit:
            gate = ROOT / "outputs/audits" / config["preflight_run_id"]
            if read(gate / "run_status.json")["status"] != "complete":
                raise ValueError("Complete preflight required")
            previous = read(gate / "input_inventory.json")
            if previous != inputs:
                raise ValueError("Preflight input/source identities changed; prepare a new gate")
            for item in read(gate / "artifact_manifest.json")["files"]:
                if sha256_file(gate / item["relative_to_run"]) != item["sha256"]:
                    raise ValueError("Preflight artifact changed")
        run.write_json("input_inventory.json", inputs)
        print("Loading cached pinned dataset and checking ordered canonical fingerprints...")
        clock = time.perf_counter()
        dataset = load_winvoker_dataset(data_config["dataset_id"], data_config["dataset_revision"])
        splits = preprocess_dataset_with_profile(dataset, profile,
                                                assignments_path=local(data_config["historic_split"]["assignments_path"]))
        del dataset
        fingerprints = fingerprint_preprocessed_splits(splits)
        verify_split_fingerprints(fingerprints, expected["splits"])
        lineage = load_winvoker_test_lineage(local(data_config["strict_test"]["mask_path"]))
        # Build a validation-only lineage receipt and validate test identity even before fitting.
        test = splits["test"]
        build_winvoker_prediction_frame(lineage, test["text"], test["label"], test["label"])
        timings = {"load_and_lineage_seconds": time.perf_counter() - clock}
        run.write_json("preprocessing.json", dict(profile_record, verified_split_fingerprints=fingerprints))
        resources = {"observed_process_rss_bytes_after_data": process_rss_bytes(),
                     "split_rows": {name: len(frame) for name, frame in splits.items()},
                     "fit_allowance_minutes": [10, 60], "host_ram_allowance_gb": [8, 16],
                     "additional_disk_allowance_gb": [2, 4], "gpu_required": False,
                     "estimate_is_not_benchmark": True,
                     "caveat": "Sparse nnz/solver runtime measured during fit; no fabricated historical hardware/version."}
        run.write_json("resources.json", resources)
        if args.preflight:
            run.write_json("results.json", {"input_and_lineage_gate": "passed", "fitting_performed": False,
                                            "inference_performed": False, "final_scientific_acceptance": False,
                                            "next_action": "Review receipt/resources before user CPU fit command"})
            run.write_text("report.md", "# SVM preflight\n\nParameters match trusted historical inventory. "
                           "Current package versions, cached dataset, split/mask and ordered cleaned-text identities checked. "
                           "No vectorizer/classifier fitted; no predictions computed. New fit is not retroactive historical provenance.\n")
            run.write_json("timings.json", timings)
            run.declare_complete(["input_inventory.json", "preprocessing.json", "resources.json",
                                  "results.json", "report.md", "timings.json"])
            return
        train = splits["train"]
        params = dict(config["vectorizer"])
        if params.pop("dtype") != {"python_type": "numpy.float64"}:
            raise ValueError("Unsupported TF-IDF dtype")
        params["dtype"] = np.float64
        params["ngram_range"] = tuple(params["ngram_range"])
        vectorizer = TfidfVectorizer(**params)
        classifier = LinearSVC(**config["classifier"])
        print("Fitting TRAIN-ONLY TF-IDF and one original-setup LinearSVC...")
        clock = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x_train = vectorizer.fit_transform(train["text"])
            classifier.fit(x_train, train["label"])
        timings["feature_and_classifier_fit_seconds"] = time.perf_counter() - clock
        # Retain fit timing even if a later artifact/evaluation step fails.
        run.write_json("timings.json", timings)
        warning_rows = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
        run.write_json("training_record.json", {
            "warnings": warning_rows, "convergence_warning": any(w["category"] == "ConvergenceWarning" for w in warning_rows),
            "classifier_n_iter": int(classifier.n_iter_), "classes": classifier.classes_.tolist(),
            "train_sparse_shape": list(x_train.shape), "train_sparse_nnz": int(x_train.nnz),
            "train_sparse_bytes": int(x_train.data.nbytes + x_train.indices.nbytes + x_train.indptr.nbytes),
            "process_rss_bytes": process_rss_bytes(), "selection": config["selection"]})
        del x_train
        # sklearn vocabulary indices can be numpy.int64, which plain JSON hashing
        # cannot encode. Normalize before BOTH writing and calculating identity.
        vocabulary = [[str(term), int(index)] for term, index in
                      sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
        run.write_json("vocabulary.json", {"fit_split": "train_only", "terms_in_index_order": vocabulary,
                                         "vocabulary_sha256": canonical_json_sha256({"terms": vocabulary}),
                                         "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest()})
        with run.path("model_bundle.pkl").open("wb") as stream:
            pickle.dump({"vectorizer": vectorizer, "model": classifier,
                         "metadata": {"run_id": run_id, "preprocessing": profile_record,
                                      "external_use": "not authorized; historical external bundle remains frozen"}}, stream)
        if not np.array_equal(classifier.classes_, [0, 1, 2]):
            raise ValueError("Noncanonical class ordering")
        for name in ("val", "test"):
            frame = splits[name]
            clock = time.perf_counter()
            x = vectorizer.transform(frame["text"])
            predicted = classifier.predict(x)
            scores = classifier.decision_function(x)
            del x
            timings[name + "_evaluation_seconds"] = time.perf_counter() - clock
            if name == "val":
                predictions = pd.DataFrame({
                    "sample_id": [stable_winvoker_sample_id(int(v)) for v in frame["source_row_index"]],
                    "source_row_index": frame["source_row_index"], "order_in_split": frame["order_in_split"],
                    "source": frame["dataset"], "text": frame["text"],
                    "cleaned_text_sha256": frame["text"].map(sha256_text),
                    "true_id": frame["label"], "predicted_id": predicted})
                for i, label in enumerate(("positive", "negative", "notr")):
                    predictions["decision_score_" + label] = scores[:, i]
                run.write_csv("validation_predictions.csv", predictions)
                validation = classification_metrics(frame["label"], predicted, subset_id="historic_validation")
                run.write_json("validation_metrics.json", validation)
            else:
                predictions = build_winvoker_prediction_frame(lineage, frame["text"], frame["label"],
                                                               predicted, scores, "decision_score")
                run.write_csv("predictions.csv", predictions)
                metrics = build_evaluation_metrics(predictions)
                run.write_json("metrics.json", metrics)
                run.write_csv("metric_views.csv", flatten_metric_views(metrics))
                run.write_csv("confusion_matrices.csv", confusion_matrix_rows(metrics))
                old = pd.read_csv(old_predictions_path, keep_default_na=False)
                if len(old) != len(predictions) or not np.array_equal(old["true_id"], frame["label"]) or not np.array_equal(old["text"], frame["text"]):
                    raise ValueError("Historical comparison row identity mismatch")
                run.write_json("historical_comparison.json", {
                    "historical_run_id": config["historical_run_id"],
                    "changed_test_predictions": int(np.sum(old["predicted_id"].to_numpy() != predicted)),
                    "old_complete": classification_metrics(old["true_id"], old["predicted_id"], subset_id="historic_test_complete"),
                    "new_complete": metrics["views"]["historic_test_complete"],
                    "interpretation": "New fit only; no claim that historical training records are repaired; external historical checkpoint unchanged."})
        for item in inputs:
            if sha256_file(local(item["path"])) != item["sha256"]:
                raise ValueError(f"Input changed during refit: {item['path']}")
        if sha256_file(historical_bundle) != config["historical_bundle_sha256"]:
            raise ValueError("Historical bundle changed")
        run.write_json("timings.json", timings)
        run.write_text("report.md", "# Complete-record SVM candidate refit\n\nCPU fit completed; inspect training_record.json "
                       "for convergence warnings before scientific acceptance. TRAIN ONLY vocabulary/IDF; explicit original "
                       "parameters and pinned current environment. Scores are SVM decision scores, not probabilities/logits. "
                       "Original artifacts, confirmatory statistics and external checkpoint remain unchanged. "
                       "No automatic manuscript promotion or public-package inclusion.\n")
        run.declare_complete(["input_inventory.json", "preprocessing.json", "resources.json", "training_record.json",
                              "vocabulary.json", "model_bundle.pkl", "validation_predictions.csv", "validation_metrics.json",
                              "predictions.csv", "metrics.json", "metric_views.csv", "confusion_matrices.csv",
                              "historical_comparison.json", "timings.json", "report.md"])


if __name__ == "__main__":
    main()
