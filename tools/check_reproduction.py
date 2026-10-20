"""Bounded recipe/prepare-CLI checks ONLY; fitting is a forbidden test boundary.

Never loads dataset/model/tokenizer/checkpoint/vector archive or calls fit.
Requires independently reconstructed input CSVs/fingerprint manifest in a NEW
execution copy. Preparation hashes files but does not load scientific examples.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not (ROOT / "workspace_receipt.json").is_file():
        raise ValueError("Use a NEW isolated execution copy")
    output = ROOT / "outputs/audits/reproduction_prepare_check_v1"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "running", "error": None}

    def write(name, payload):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write("run_status.json", status)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="")
    sys.path.insert(0, str(ROOT))
    attempts = []

    def audit(event, arguments):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Network disabled in reproduction preparation checks")

    sys.addaudithook(audit)
    try:
        from src.reproduction import SETUPS, resolve_setup, verify_inputs
        runner = importlib.import_module("scripts.run_reproduction")
        if not Path(runner.__file__).resolve().is_relative_to(ROOT):
            raise ValueError("Imported runner from outside isolated copy")
        forbidden_fit_calls = []

        def forbidden_fit(*args, **kwargs):
            forbidden_fit_calls.append(True)
            raise AssertionError("Fit boundary reached during preparation check")

        runner.execute_fit = forbidden_fit
        resolved, prepared, rejected = [], [], []
        for setup in sorted(SETUPS):
            seed = None if setup in {"tfidf_logreg", "tfidf_svm", "tfidf_nb", "fasttext_logreg"} else 42
            for stage in ("validation_only", "replication"):
                run_id = f"repro_check_{setup}_{stage}"
                config = resolve_setup(setup, run_id=run_id, seed=seed, stage=stage)
                has_mask = any(name.endswith("test_inclusion_mask.csv") for name in config["input_identities"])
                if has_mask != (stage == "replication"):
                    raise ValueError("Held-out mask access in validation-only preparation or absent in replication")
                if config["training_seed"] != seed or config["original_acceptance_recreated"]:
                    raise ValueError("Replication/singleton identity differs")
                resolved.append({"setup": setup, "stage": stage, "training_seed": seed,
                                 "pooling": config["model_config"]["project_head"]["pooling"] if config["model_config"] else None})
                if setup != "fasttext_logreg":
                    args = ["--setup", setup, "--run-id", run_id, "--stage", stage]
                    if seed is not None:
                        args += ["--seed", str(seed)]
                    runner.main(args)  # mode defaults to preparation, real reconstructed hashes checked
                    saved = ROOT / "outputs/preparations" / run_id / "resolved_config.json"
                    if json.loads(saved.read_text(encoding="utf-8")) != config:
                        raise ValueError("Prepared configuration serialization differs")
                    prepared.append(run_id)
        invalid = [
            ("historical_id", {"name": "tfidf_nb", "run_id": "s1_tfidf_nb_original"}),
            ("singleton_seed", {"name": "fasttext_logreg", "run_id": "repro_invalid", "seed": 42}),
            ("unlocked_seed", {"name": "bilstm", "run_id": "repro_invalid", "seed": 43}),
            ("classical_cuda", {"name": "tfidf_nb", "run_id": "repro_invalid", "device": "cuda"}),
        ]
        for label, arguments in invalid:
            try:
                resolve_setup(**arguments)
            except ValueError:
                rejected.append(label)
            else:
                raise ValueError(f"Invalid recipe request accepted: {label}")
        for label, tail in (("fit_without_confirmation", ["--mode", "fit"]),
                            ("confirmation_without_fit", ["--confirm-new-fit"])):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    runner.main(["--setup", "tfidf_nb", "--run-id", "repro_invalid", *tail])
            except SystemExit as error:
                if error.code != 2:
                    raise
                rejected.append(label)
            else:
                raise ValueError("Fit confirmation guard did not reject invalid invocation")
        try:
            verify_inputs(resolve_setup("fasttext_logreg", run_id="repro_missing_vectors"))
        except ValueError:
            rejected.append("fasttext_missing_archive")
        else:
            raise ValueError("fastText preparation did not require separately acquired archive")
        # Structural test access check, not an executed model branch.
        tree = ast.parse((ROOT / "scripts/run_reproduction.py").read_text(encoding="utf-8"))
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_fit")
        guarded = [node for node in ast.walk(execute) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "config['stage'] == 'replication'"]
        if len(guarded) != 1:
            raise ValueError("Expected one explicit held-out replication branch")
        branch = guarded[0]
        protected_calls = {id(node) for node in ast.walk(branch) if isinstance(node, ast.Call)}
        for node in ast.walk(execute):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"load_winvoker_test_lineage", "build_winvoker_prediction_frame", "build_evaluation_metrics"}:
                if id(node) not in protected_calls:
                    raise ValueError("Held-out lineage/prediction/metrics call outside replication guard")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "preprocess_dataset_with_profile":
                for keyword in node.keywords:
                    if keyword.arg == "included_splits" and any(isinstance(item, ast.Constant) and item.value == "test" for item in ast.walk(keyword.value)):
                        if id(node) not in protected_calls:
                            raise ValueError("Held-out text construction outside replication guard")
        if forbidden_fit_calls or attempts or (ROOT / "outputs/experiments").exists():
            raise ValueError("Forbidden fitting/network/experiment outputs reached")
        write("results.json", {"status": "recipe_resolution_and_prepare_dispatch_passed",
                              "resolved_cases": resolved, "prepare_cli_runs": prepared,
                              "rejected_invalid_cases": rejected, "held_out_calls_structurally_guarded": True,
                              "fit_boundary_calls": 0, "network_attempts": 0, "fitting_or_inference": False,
                              "dataset_or_model_loading": False, "gpu": False,
                              "fasttext_archive_loading_tested": False,
                              "actual_fit_branches_executed": False,
                              "limitation": "Resolution and prepare paths only; no model-fit numerical parity or clean-install proof"})
        status["status"] = "complete"
        print(f"Resolved {len(resolved)} cases; {len(prepared)} prepare-only CLI runs; no fitting/inference.")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write("run_status.json", status)


if __name__ == "__main__":
    main()
