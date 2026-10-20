"""User-run trusted historical estimator inventory; no fitting or inference.

Pickle is unsafe for arbitrary inputs. This entrypoint ONLY deserializes the two
local historical bundles after matching previously accepted SHA-256 identities.
Archive byte identity is deferred to the fit gate; this cheap inventory only
checks archive presence/size. Outputs remain research-only, never repository/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import re
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUSTED = {
    "svm": ("s1_tfidf_svm_winvoker_seed42_base",
            "0e381fcc40572297b9e792bc2d5cec8eb224dcfdc712feb75455d23a266677d0"),
    "fasttext_logreg": ("s2_embed_fasttext_logreg_winvoker_seed42_base",
                       "e2d46e3c8588d3265d0333d85b70118a9d3ca2f386831ec9c3af085711b3a228"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def local(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe input: {name}")
    result = ROOT / path
    result.resolve().relative_to(ROOT.resolve())
    if result.is_symlink():
        raise ValueError(f"Symlink input refused: {name}")
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def parameter(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [parameter(v) for v in value]
    if isinstance(value, dict):
        return {str(k): parameter(v) for k, v in value.items()}
    if isinstance(value, type):
        return {"python_type": f"{value.__module__}.{value.__qualname__}"}
    raise TypeError(f"Unsupported estimator parameter type: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = local(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "cpu-refit-inventory-v1":
        raise ValueError("Unsupported config")
    if any(config.get(k) is not False for k in ("training", "inference", "network")):
        raise ValueError("Inventory-only contract required")
    if set(config["models"]) != set(TRUSTED):
        raise ValueError("Unexpected model scope")
    run_id = config["run_id"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    output = ROOT / "outputs" / "audits" / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"run_id": run_id, "status": "running", "started_utc": now(),
              "training": False, "inference": False, "network": False}
    write(output / "run_status.json", status)
    try:
        write(output / "config.json", config)
        packages = {}
        for name in ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "threadpoolctl"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            ram = None
        write(output / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "logical_cpu_count": os.cpu_count(), "visible_ram_bytes": ram,
            "output_disk_free_bytes": shutil.disk_usage(output).free,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "hardware_scope": "container-visible; not a fit benchmark or guaranteed free RAM",
        })
        inputs = []
        for name in [args.config, "scripts/audit_cpu_refit_inventory.py", *config["additional_inputs"]]:
            path = local(name)
            inputs.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
        results = {}
        for slot, spec in config["models"].items():
            if (spec["historical_run_id"], spec["bundle_sha256"]) != TRUSTED[slot]:
                raise ValueError(f"Untrusted bundle contract: {slot}")
            folder = "outputs/experiments/" + spec["historical_run_id"]
            bundle_path = local(folder + "/model_bundle.pkl")
            # Load the exact bytes hashed, not a second potentially changed file.
            payload = bundle_path.read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            if observed != spec["bundle_sha256"]:
                raise ValueError(f"Historical bundle identity mismatch: {slot}; refusing pickle")
            inputs.append({"path": folder + "/model_bundle.pkl", "sha256": observed,
                           "bytes": len(payload)})
            historical_path = local(folder + "/config.json")
            inputs.append({"path": folder + "/config.json", "sha256": sha(historical_path),
                           "bytes": historical_path.stat().st_size})
            historical = json.loads(historical_path.read_text(encoding="utf-8"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bundle = pickle.loads(payload)
            records = {}
            for key, expected in spec["estimators"].items():
                estimator = bundle[key]
                identity = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
                if identity != expected:
                    raise ValueError(f"Unexpected estimator identity: {slot}/{key}: {identity}")
                records[key] = {"class": identity,
                                "parameters": parameter(estimator.get_params(deep=False)),
                                "classes": parameter(estimator.classes_.tolist())
                                if hasattr(estimator, "classes_") else None}
            results[slot] = {
                "historical_run_id": spec["historical_run_id"], "historical_config": historical,
                "bundle_keys": sorted(bundle), "estimators": records,
                "load_warnings": [{"category": type(w.message).__name__, "message": str(w.message),
                                   "original_sklearn_version": getattr(w.message, "original_sklearn_version", None)}
                                  for w in caught],
                "historical_version_note": "Warnings can identify a version mismatch; absence does not prove historical version.",
                "fit_ready": False, "inference_performed": False,
            }
        archive = local(config["fasttext_archive"])
        archive_state = {"path": config["fasttext_archive"], "present": archive.is_file(),
                         "bytes": archive.stat().st_size if archive.is_file() else None,
                         "accepted_sha256": config["accepted_archive_sha256"],
                         "sha256_rechecked": False,
                         "identity_gate": "Full archive SHA-256 required before the future fit; not verified here."}
        archive_state["size_matches_accepted"] = archive_state["bytes"] == config["accepted_archive_size_bytes"]
        write(output / "input_inventory.json", inputs)
        write(output / "results.json", {"models": results, "fasttext_archive": archive_state,
                                        "fit_ready": False, "scientific_acceptance": False})
        (output / "report.md").write_text(
            "# CPU refit parameter inventory\n\nTrusted historical bundles inspected; no fit, prediction, "
            "dataset loading or network access. Review results.json for all parameters and load warnings.\n\n"
            "This gate is not fit acceptance or historical-environment reconstruction. Reconcile defaults/versions, "
            "freeze executable configurations and refine resources before fit commands. FastText remains one "
            "deterministic configuration; random_state is provenance, not a seed sweep. External bundles remain frozen. "
            "Archive size is not byte identity.\n", encoding="utf-8")
        status.update(status="complete", completed_utc=now(), elapsed_seconds=time.perf_counter() - started)
        write(output / "run_status.json", status)
        write(output / "artifact_manifest.json", {"run_id": run_id, "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(output.iterdir()) if p.is_file()]})
        print(f"Run completed: {run_id}; output: {output.relative_to(ROOT)}; no fitting/inference.")
    except Exception as exc:
        status.update(status="failed", failed_utc=now(), error=str(exc),
                      elapsed_seconds=time.perf_counter() - started)
        write(output / "run_status.json", status)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
