"""Pure standard-library resolution for NEW replications, not historical approval.

Importing/resolving this module cannot load data/models, fit or infer. Historical
runners and acceptance gates are deliberately not modified or monkeypatched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "configs/reproduction/accepted_setups_v1.json"
SETUPS = {"tfidf_logreg", "tfidf_svm", "tfidf_nb", "tfidf_rf", "fasttext_logreg", "bilstm",
          "berturk_standard_ce", "berturk_weighted_ce", "mbert_standard_ce", "xlm_roberta_standard_ce",
          "turkish_electra_standard_ce", "distilberturk_standard_ce", "convberturk_standard_ce"}
CANONICAL_SHA = "ab69f95592e93a9afce3709dfa88d7dd5722afab556e262f353a6265246b8561"
FINGERPRINT_SHA = "824afeeb30e18d60f8f752df4a2b9424d19cab62018fe400f2cddb878115879c"
ARCHIVE_SHA = "b0c1a9ca6da5e2bc5285f3a1be4a1755b1721e3192e7698a25f2c39bb28c5fea"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Package path escapes execution workspace")
    return path


def resolve_setup(name, *, run_id, seed=None, stage="validation_only", device="cpu", archive=None):
    """Resolve one fixed recipe; does not read dataset/weights or launch a runner."""
    if not re.fullmatch(r"repro_[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or ".." in run_id:
        raise ValueError("New replications require a fresh path-safe repro_ ID; historical IDs forbidden")
    if stage not in {"validation_only", "replication"} or device not in {"cpu", "cuda"}:
        raise ValueError("Explicit validation_only/replication stage and cpu/cuda device required")
    recipes = read(RECIPES)
    if recipes.get("schema_version") != "code-only-reproduction-recipes-v1" or set(recipes["setups"]) != SETUPS:
        raise ValueError("Unexpected fixed recipe schema/membership")
    if name not in SETUPS or recipes["split_seed"] != 42:
        raise ValueError("Unknown recipe or changed split seed")
    setup = copy.deepcopy(recipes["setups"][name])
    stochastic = setup["replication"] == "five_locked_training_seeds"
    if stochastic:
        if isinstance(seed, bool) or seed not in (42, 52, 62, 72, 82):
            raise ValueError("This stochastic setup requires one of the five locked training seeds")
    elif seed is not None:
        raise ValueError("Singleton configurations do not accept a training seed")
    if setup["family"] in {"classical", "fasttext"} and device != "cpu":
        raise ValueError("Classical/fastText configurations are CPU-only")
    family = setup["family"]
    parameters = {}
    if family in {"bilstm", "transformer"}:
        parameters.update(copy.deepcopy(recipes["neural_common"]))
    if family == "transformer":
        parameters.update(copy.deepcopy(recipes["transformer_common"]))
    if family == "classical":
        parameters["tfidf_constructor"] = copy.deepcopy(recipes["tfidf_constructor"])
    parameters.update(setup)
    if name == "tfidf_rf":
        parameters["constructor_arguments"]["random_state"] = seed
    dataset = read(local(recipes["dataset_config"]))
    if (dataset["dataset_id"] != "winvoker/turkish-sentiment-analysis-dataset"
            or dataset["dataset_revision"] != "4bd9103ad5f24d44db806d40c94c8b9ec116ad05"):
        raise ValueError("Mutable or changed dataset revision")
    if sha(local(recipes["preprocessing_profile"])) != CANONICAL_SHA:
        raise ValueError("Canonical preprocessing configuration changed")
    model = None
    if family == "transformer":
        model = read(local(setup["model_config"]))
        for field in ("model_revision", "tokenizer_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", str(model.get(field, ""))):
                raise ValueError("Model/tokenizer must have immutable full revisions")
        head = model["project_head"]
        if head["classifier"] != "linear" or head["num_classes"] != 3 or head["pooling"] not in {"pooler_output", "first_token"}:
            raise ValueError("Unexpected reported final model head")
        if parameters["max_length"] > model["maximum_position_embeddings"]:
            raise ValueError("Requested sequence length exceeds the pinned backbone limit")
    inputs = {
        dataset["historic_split"]["assignments_path"]: dataset["historic_split"]["assignments_sha256"],
        recipes["preprocessing_fingerprints"]: FINGERPRINT_SHA,
    }
    # Validation-only preparation/fitting never reads the held-out mask.
    if stage == "replication":
        inputs[dataset["strict_test"]["mask_path"]] = dataset["strict_test"]["mask_sha256"]
    sources = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), RECIPES,
                      local(recipes["dataset_config"]), local(recipes["preprocessing_profile"])])
    if model is not None:
        sources.append(local(setup["model_config"]))
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    return {
        "schema_version": "independent-replication-config-v1", "run_id": run_id,
        "setup": name, "family": family, "stage": stage, "device": device,
        "training_seed": seed if stochastic else None,
        "implementation_random_state": seed if stochastic else 42,
        "parameters": parameters, "model_config": model,
        "dataset_config": recipes["dataset_config"],
        "preprocessing_profile": recipes["preprocessing_profile"],
        "preprocessing_fingerprints": recipes["preprocessing_fingerprints"],
        "input_identities": inputs, "source_hashes": source_hashes,
        "fasttext_archive": str(Path(archive).resolve()) if archive else None,
        "original_acceptance_recreated": False,
        "historical_runs_or_external_roster_replaced": False,
        "configuration_scope": "new fixed-setup replication, not original fit identity or original approval",
        "test_access": "none" if stage == "validation_only" else "only after fitting/validation checkpoint selection",
    }


def verify_inputs(config):
    for name, expected in config["input_identities"].items():
        if sha(local(name)) != expected:
            raise ValueError(f"Reconstructed input identity differs: {name}")
    if config["family"] == "fasttext":
        if not config["fasttext_archive"] or sha(config["fasttext_archive"]) != ARCHIVE_SHA:
            raise ValueError("Separately acquired local fastText archive is required with the recorded content identity")
