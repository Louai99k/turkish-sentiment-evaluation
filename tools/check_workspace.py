"""Bounded offline imports and synthetic function checks; no model/data loading.

Run in an isolated workspace. This is NOT a training or result-reproduction test.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (ROOT / "workspace_receipt.json").is_file():
        raise ValueError("Run only in a NEW isolated execution workspace")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Check receipts must remain in this execution workspace")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "running", "error": None}

    def write(name, payload):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write("run_status.json", status)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
    # Python -I suppresses inherited PYTHONPATH; insert only this execution copy.
    sys.path.insert(0, str(ROOT))
    network_attempts = []

    def audit(event, arguments):
        if event == "socket.connect":
            network_attempts.append(event)
            raise RuntimeError("Network disabled during bounded workspace checks")

    sys.addaudithook(audit)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        def forbidden_load(*args, **kwargs):
            raise RuntimeError("Checkpoint/tokenizer loading forbidden during bounded workspace checks")

        # No model/Trainer/estimator is instantiated, and no runner main() called.
        AutoModel.from_pretrained = forbidden_load
        AutoTokenizer.from_pretrained = forbidden_load
        torch.load = forbidden_load
        modules = (
            "src.protocol_constants", "src.preprocessing", "src.protocol_predictions",
            "src.protocol_metrics", "src.data_loader", "src.model", "src.transformer_model",
            "src.train", "scripts.run_transformer_experiment", "scripts.run_bilstm_protocol_v2",
            "scripts.run_fasttext_train_vocab_refit", "scripts.run_svm_complete_records",
        )
        imported = {}
        for name in modules:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f"Imported private/outside project source: {name}")
            imported[name] = path.relative_to(ROOT).as_posix()
        from src.data_loader import clean_text
        from src.preprocessing import (
            apply_preprocessing_profile, load_preprocessing_profile, ordered_cleaned_text_sha256,
            preprocessing_profile_record, validate_preprocessing_profile,
        )
        from src.protocol_metrics import classification_metrics

        profile_path, profile = load_preprocessing_profile("configs/preprocessing/winvoker_canonical_v1.json")
        record = preprocessing_profile_record(profile_path, profile)
        expected_pin = "ab69f95592e93a9afce3709dfa88d7dd5722afab556e262f353a6265246b8561"
        if record["profile_config_sha256"] != expected_pin:
            raise ValueError("Canonical profile bytes differ")
        probes = ["", "Merhaba!", "  <b>İYİ</b> @kisi https://example.test !! ",
                  "i\u0307", "www.example.test kötü?", "🙂 harika", "... !!! ???", "A\n B\tC"]
        for text in probes:
            if apply_preprocessing_profile(text, profile) != clean_text(text):
                raise ValueError("Profile/legacy cleaning mismatch on synthetic probe")
        if apply_preprocessing_profile(probes[2], profile) != "İYİ !!":
            raise ValueError("Canonical HTML/URL/mention/casing contract differs")
        if apply_preprocessing_profile("i\u0307", profile) != "i\u0307":
            raise ValueError("Canonical combining-dot behavior differs")
        if ordered_cleaned_text_sha256(["a", "bc"]) == ordered_cleaned_text_sha256(["ab", "c"]):
            raise ValueError("Fingerprint fails record-boundary check")
        if ordered_cleaned_text_sha256(["a", "b"]) == ordered_cleaned_text_sha256(["b", "a"]):
            raise ValueError("Fingerprint fails ordering check")
        invalid = copy.deepcopy(profile)
        invalid["ordered_transformations"][0]["order"] = 99
        try:
            validate_preprocessing_profile(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid transformation ordering was accepted")
        perfect = classification_metrics([0, 1, 2], [0, 1, 2], subset_id="synthetic_perfect")
        if perfect["accuracy"] != 1.0 or perfect["macro_f1"] != 1.0:
            raise ValueError("Synthetic perfect metric check failed")
        example = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], subset_id="synthetic_confusion")
        if example["confusion_matrix"] != [[1, 1, 0], [0, 2, 0], [1, 0, 1]] or abs(example["accuracy"] - 4 / 6) > 1e-12:
            raise ValueError("Synthetic confusion check failed")
        if network_attempts:
            raise ValueError("A network attempt occurred, even if caught by an imported library")
        versions = {}
        for package in ("torch", "transformers", "datasets", "huggingface-hub", "scikit-learn",
                        "numpy", "pandas", "scipy", "matplotlib", "seaborn", "regex", "tqdm"):
            versions[package] = importlib.metadata.version(package)
        write("results.json", {
            "status": "bounded_import_and_synthetic_checks_passed",
            "local_imports": imported, "profile_identity_passed": True,
            "synthetic_preprocessing_probes": len(probes), "fingerprint_checks_passed": True,
            "invalid_profile_rejected": True, "synthetic_metric_checks_passed": True,
            "network_attempts": len(network_attempts), "model_or_tokenizer_loaded": False,
            "dataset_loaded": False, "fitting_or_inference": False, "gpu_used": False,
            "packages": versions, "python": sys.version,
            "limitation": "Existing installed environment and selected module imports only; not a clean dependency installation or end-to-end scientific reproduction"})
        status["status"] = "complete"
        print("Bounded workspace imports and synthetic checks passed; no data/model loading or fitting.")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write("run_status.json", status)


if __name__ == "__main__":
    main()
