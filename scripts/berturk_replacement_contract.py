"""Immutable preparation gate helpers; no model loading or training imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "outputs/audits/professor_round1_berturk_replacement_plan_v1"
MATRIX = ROOT / "configs/final/berturk_provenance_replacements_v1.json"
PINS = {
    "matrix": "6c9b79c14920d787b01c16bf5b71fb65b91b61e7e382af5bd5fce02579ec3c89",
    "report.json": "1c0932606ec84dc91ad40593472973a90a7c6a68c74178cfbc43a0bd8cc55c8f",
    "resolved_run_configs.json": "1d4bbb83045c44c05578340456060fc2521ef7c1e99de2b3782122696605dbc7",
    "input_inventory.json": "14df5a7e1458d64804a454414fc0e382a6bcc633787076c6b31ba5150184faa9",
    "cache_inventory.json": "88d712d4354ae1e34034fb3a08c1d6e3beea482bdb1507714df892e338b97e43",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def preparation_gate():
    status = read(PREPARATION / "run_status.json")
    require(status["run_id"] == PREPARATION.name and status["status"] == "complete" and
            status["error"] is None, "Accepted preparation incomplete")
    manifest = read(PREPARATION / "artifact_manifest.json")
    require(manifest["run_id"] == PREPARATION.name and manifest["status"] == "complete", "Wrong preparation manifest")
    seen = set()
    for entry in manifest["files"]:
        name = entry["name"]
        require(Path(name).name == name and name not in seen, "Unsafe/duplicate manifest path")
        seen.add(name)
        path = PREPARATION / name
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"], "Preparation payload changed")
    require(sha(MATRIX) == PINS["matrix"], "Frozen replacement matrix changed")
    for name, pin in PINS.items():
        if name != "matrix":
            require(sha(PREPARATION / name) == pin, f"Accepted preparation pin changed: {name}")
    preservation = read(PREPARATION / "preservation_check.json")
    require(preservation["inputs_unchanged"] and preservation["cache_files_unchanged"], "Preparation preservation failed")
    matrix = read(MATRIX)
    configs = read(PREPARATION / "resolved_run_configs.json")
    require(len(configs) == 6 and {r["run_id"] for r in configs} ==
            {r["new_id"] for r in matrix["replacements"]}, "Six configuration identities changed")
    for row in configs:
        require(canonical_sha(row["resolved_config"]) == row["canonical_config_sha256"], "Canonical run config changed")
    before = read(PREPARATION / "input_inventory.json")
    print("Checking365 accepted preparation inputs; no model loaded yet", flush=True)
    for name, pin in before.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha(path) == pin, f"Accepted input changed: {name}")
    cache = read(PREPARATION / "cache_inventory.json")["files"]
    for row in cache:
        require(sha(Path(row["path"])) == row["sha256"], "Pinned cached snapshot file changed")
    for path in [*PREPARATION.iterdir(), Path(__file__).resolve()]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    return matrix, configs, before, cache
