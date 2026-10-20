"""Strict single-run replacement entrypoint. Model imports occur ONLY after guards."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.berturk_replacement_execution_contract import (
    ACCEPTANCE, PINS, canonical_sha, execution_acceptance_gate,
    read, require, scientific_gate, sha,
)


def validate_run_state(target, resume, identity):
    require(not target.is_symlink(), "Symlink experiment target refused")
    if not resume:
        require(not target.exists(), "Existing target refused; no overwrite or implicit restart")
        return "fresh"
    require(target.is_dir(), "Resume requires existing experiment; no epoch1 fallback")
    require((target / "run_status.json").is_file() and (target / "config.json").is_file(),
            "Resume requires recorded status/config")
    require(not (target / "run_status.json").is_symlink() and not (target / "config.json").is_symlink(),
            "Symlink resume status/config refused")
    status = read(target / "run_status.json")
    require(status["status"] in ("running", "failed", "interrupted"), "Completed/unknown run cannot resume")
    prior = read(target / "config.json")
    require(prior.get("revision_provenance", {}).get("execution_identity") == identity,
            "Resume identity changed; matrix/config/source/run must match")
    checkpoint = target / "checkpoints/resume_checkpoint.pt"
    require(not (target / "checkpoints").is_symlink() and
            checkpoint.resolve().is_relative_to(target.resolve()) and
            checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0,
            "Missing epoch-boundary resume checkpoint; epoch1 fallback forbidden")
    return "resume_existing_epoch_boundary"


def runner_arguments(config, resume=False):
    args = ["--model-config", config["model_config"], "--preprocessing-profile", config["preprocessing_profile"],
        "--preprocessing-fingerprints", config["preprocessing_fingerprints"],
        "--preprocessing-freeze-manifest", config["preprocessing_freeze_manifest"],
        "--exp-id", config["new_id"], "--seed", str(config["seed"]), "--loss-mode", config["loss_mode"],
        "--experiment-stage", config["experiment_stage"], "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]), "--learning-rate", str(config["learning_rate"]),
        "--max-length", str(config["max_length"]), "--dropout", str(config["dropout"]),
        "--early-stopping-patience", str(config["early_stopping_patience"]), "--device", "cuda", "--local-files-only"]
    return [*args, "--resume"] if resume else args


def validate_environment(seed):
    require(os.environ.get("PYTHONHASHSEED") == str(seed), "Set PYTHONHASHSEED at process startup to selected seed")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS determinism environment missing")
    require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("HF_DATASETS_OFFLINE") == "1",
            "Replacement execution requires offline pinned caches")


def make_bound_writer(base, provenance, before, cache):
    class BoundExperimentRun(base):
        def __init__(self, *values, **kwargs):
            kwargs["config"]["revision_provenance"] = provenance
            super().__init__(*values, **kwargs)

        def __enter__(self):
            result = super().__enter__()
            self.write_json("revision_contract.json", provenance)
            self.write_json("revision_input_inventory.json", before)
            return result

        def declare_complete(self, required_files):
            self._preserve()
            super().declare_complete([*required_files, "revision_contract.json",
                                      "revision_input_inventory.json", "revision_preservation_check.json"])

        def _preserve(self):
            unchanged = all(sha(ROOT / name) == pin for name, pin in before.items())
            cache_unchanged = all(sha(Path(row["path"])) == row["sha256"] for row in cache)
            self.write_json("revision_preservation_check.json", {"inputs_unchanged": unchanged,
                "checked_files": len(before), "cache_files_unchanged": cache_unchanged,
                "cache_files_checked": len(cache)})
            require(unchanged and cache_unchanged, "Original/source/cache preservation failed")

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                self._preserve()
            except BaseException:
                error = sys.exc_info()
                super().__exit__(*error)
                raise
            return super().__exit__(exc_type, exc_value, traceback)
    return BoundExperimentRun


def make_bound_trainer(base, identity):
    class BoundTrainer(base):
        def __init__(self, *values, **kwargs):
            metadata = dict(kwargs.get("checkpoint_metadata") or {})
            metadata["revision_execution_identity"] = identity
            kwargs["checkpoint_metadata"] = metadata
            super().__init__(*values, **kwargs)
    return BoundTrainer


def acquire_lock(path, run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Replacement execution lock exists; inspect live/stale owner, no automatic unlock") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "pid": os.getpid(), "owner_token": token}, stream)
    return token


def release_owned_lock(path, token):
    require(path.is_file() and read(path)["owner_token"] == token, "Lock ownership changed; do not delete")
    path.unlink()  # Only the temporary lock exclusively created by this invocation.


def execute_backend(args, provenance, before, cache):
    # Bind artifact/checkpoint identity only; no model/data/training-method changes.
    from src.experiment_artifacts import ExperimentRun
    from scripts import run_transformer_experiment as backend

    original_writer = backend.ExperimentRun
    original_trainer = backend.Trainer
    backend.ExperimentRun = make_bound_writer(ExperimentRun, provenance, before, cache)
    backend.Trainer = make_bound_trainer(original_trainer, provenance["execution_identity"])
    try:
        backend.main(args)
    finally:
        backend.ExperimentRun = original_writer
        backend.Trainer = original_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    matrix, configs, before, cache, sources = scientific_gate()
    matches = [r for r in configs if r["run_id"] == args.run_id]
    require(len(matches) == 1, "Run ID outside authorized six-fit matrix")
    row = matches[0]
    require(canonical_sha(row["resolved_config"]) == row["canonical_config_sha256"], "Resolved run config differs")
    acceptance, directory = execution_acceptance_gate(sources)
    validate_environment(row["resolved_config"]["seed"])
    target = ROOT / "outputs/experiments" / args.run_id
    identity = {"run_id": args.run_id, "matrix_sha256": PINS["matrix"],
        "canonical_config_sha256": row["canonical_config_sha256"], "source_hashes": sources,
        "execution_report_sha256": acceptance["report_sha256"]}
    validate_run_state(target, args.resume, identity)
    for path in [ACCEPTANCE, *directory.iterdir()]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    provenance = {"schema_version": "berturk-replacement-execution-v1", "execution_identity": identity,
        "historical_run_preserved": row["resolved_config"]["historical_id"],
        "resolved_scientific_config": row["resolved_config"], "scope": matrix["scope"],
        "namespace_migration": "backbone keys; pooler_output/linear computation compatibility accepted",
        "frozen_external_roster_unchanged": True, "original_confirmatory_analysis_unchanged": True}
    lock = ROOT / "outputs/audits/berturk_replacement_locks/active.lock.json"
    token = acquire_lock(lock, args.run_id)
    try:
        execute_backend(runner_arguments(row["resolved_config"], args.resume), provenance, before, cache)
    finally:
        release_owned_lock(lock, token)


if __name__ == "__main__":
    main()
