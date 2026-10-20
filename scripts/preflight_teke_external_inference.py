"""Check frozen Teke mask and model artifacts before any external inference.

Standard-library only. Does not import, load, fit, or evaluate a model and does
not read external labels for model selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/teke_external_inference_preflight_v1"
MASK_DIR = ROOT / "outputs/audits/teke_external_final_mask_v1"
SOURCE = ROOT / "data/external/teke_trendyol_original.zip"
LABELS = ("Positive", "Negative", "Notr")
SOURCE_SHA = "e2f1bc8ab44a6b23785c87c557a5df0d9cf5886d4198531807bcd626bbefb7b2"
PROFILE_SHA = "ab69f95592e93a9afce3709dfa88d7dd5722afab556e262f353a6265246b8561"
MASK_SHA = "71bb85218868c2908257c5783f12b129a1693e640f7e0d27bde2ae728a4541a2"
SUMMARY_SHA = "8c0749f71192d6e54f3daa8ce0e1bb7e1ec499b7dd650e8f25c59a0db6ddab7b"

ROSTER = (
    {
        "slot": "berturk",
        "run": "abl_berturk_winvoker_seed52_unweighted_valf1_v2",
        "artifact": "checkpoints/best_checkpoint.pt",
        "expected_sha256": "97a4e64ff12650046d8d983b603584ba56877700af9305f0690adeba906a890f",
        "validation_weighted_f1": 0.9718677345333187,
        "model_id": "dbmdz/bert-base-turkish-cased",
        "model_revision": "b6e1de16c983e0f2c70664591ea3f22810072608",
        "seed": 52,
    },
    {
        "slot": "xlm_roberta",
        "run": "final_xlm_roberta_winvoker_canonical_len128_seed42_unweighted_v1",
        "artifact": "checkpoints/best_checkpoint.pt",
        "expected_sha256": "c82499114c16cc38c683b18310b9f63c75897bc64346a73b58930cf30b291243",
        "validation_weighted_f1": 0.9684325368549945,
        "model_id": "FacebookAI/xlm-roberta-base",
        "model_revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        "seed": 42,
    },
    {
        "slot": "bilstm",
        "run": "s2_neural_bilstm_winvoker_seed52_protocol_v2",
        "artifact": "checkpoints/best_checkpoint.pt",
        "expected_sha256": "b20b7c921a45f2d64d63c1cb816f3509ab3afe7f340377a9261f772a53488d70",
        "validation_weighted_f1": 0.9327610909544053,
        "seed": 52,
    },
    {
        "slot": "linear_svm",
        "run": "s1_tfidf_svm_winvoker_seed42_base",
        "artifact": "model_bundle.pkl",
        "expected_sha256": None,
        "validation_weighted_f1": 0.9277682125871497,
        "seed": 42,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def manifest_hash(manifest: dict, relative: str) -> str:
    matches = [entry for entry in manifest["files"] if entry["relative_to_run"] == relative]
    require(len(matches) == 1, f"Manifest lacks unique {relative}")
    return matches[0]["sha256"]


def check_mask() -> dict:
    require(sha256(SOURCE) == SOURCE_SHA, "Teke original ZIP hash changed")
    manifest = read_json(MASK_DIR / "manifest.json")
    require(manifest["summary.json"] == SUMMARY_SHA, "Frozen summary manifest hash changed")
    require(manifest["final_mask.csv"] == MASK_SHA, "Frozen mask manifest hash changed")
    require(sha256(MASK_DIR / "summary.json") == SUMMARY_SHA, "Frozen summary content changed")
    require(sha256(MASK_DIR / "final_mask.csv") == MASK_SHA, "Frozen mask content changed")
    require(read_json(MASK_DIR / "run_status.json")["status"] == "complete", "Mask run incomplete")
    summary = read_json(MASK_DIR / "summary.json")
    require(summary["source_zip_sha256"] == SOURCE_SHA, "Mask source lineage mismatch")
    expected_views = {
        "primary_included": (67964, {"Positive": 56429, "Negative": 10777, "Notr": 758}),
        "short_sensitivity_included": (67223, {"Positive": 55900, "Negative": 10639, "Notr": 684}),
    }
    for name, (count, support) in expected_views.items():
        require(summary["views"][name] == {"rows": count, "support": support}, f"Unexpected {name} support")
    observed = {name: Counter() for name in expected_views}
    seen = set()
    with (MASK_DIR / "final_mask.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "label", "final_primary_included", "short_text_sensitivity_included"}
        require(required.issubset(reader.fieldnames or ()), "Mask columns missing")
        for row in reader:
            sample_id = row["sample_id"]
            require(sample_id not in seen, f"Duplicate mask sample ID: {sample_id}")
            seen.add(sample_id)
            require(row["label"] in LABELS, f"Unexpected mask label: {row['label']}")
            primary = row["final_primary_included"]
            short = row["short_text_sensitivity_included"]
            require(primary in ("True", "False") and short in ("True", "False"), "Invalid mask flag")
            require(short != "True" or primary == "True", "Sensitivity row outside primary mask")
            if primary == "True":
                observed["primary_included"][row["label"]] += 1
            if short == "True":
                observed["short_sensitivity_included"][row["label"]] += 1
    for name, (count, support) in expected_views.items():
        require(sum(observed[name].values()) == count and dict(observed[name]) == support,
                f"Mask row-level {name} support mismatch")
    return {"source_zip_sha256": SOURCE_SHA, "final_mask_sha256": MASK_SHA,
            "summary_sha256": SUMMARY_SHA, "mask_rows": len(seen), "views": summary["views"]}


def check_artifact(item: dict) -> dict:
    run_dir = ROOT / "outputs/experiments" / item["run"]
    artifact = run_dir / item["artifact"]
    observed_sha = sha256(artifact)
    if item["expected_sha256"] is not None:
        require(observed_sha == item["expected_sha256"], f"{item['slot']} artifact hash changed")
        require(read_json(run_dir / "run_status.json")["status"] == "complete",
                f"{item['slot']} source run incomplete")
        manifest = read_json(run_dir / "artifact_manifest.json")
        require(manifest["run_status"] == "complete" and manifest_hash(manifest, item["artifact"]) == observed_sha,
                f"{item['slot']} source manifest mismatch")
        metadata = read_json(run_dir / "checkpoint_metadata.json")
        require(manifest_hash(manifest, "checkpoint_metadata.json") == sha256(run_dir / "checkpoint_metadata.json"),
                f"{item['slot']} checkpoint metadata hash mismatch")
        require(abs(metadata["validation_weighted_f1"] - item["validation_weighted_f1"]) < 1e-12,
                f"{item['slot']} validation selection value mismatch")
        if item["slot"] == "bilstm":
            require(metadata["sha256"] == observed_sha and
                    metadata["run_identity"]["profile_config_sha256"] == PROFILE_SHA,
                    "BiLSTM checkpoint/profile mismatch")
        else:
            model = metadata["metadata"]
            require(model["model_id"] == item["model_id"] and
                    model["model_revision"] == item["model_revision"] and
                    model["tokenizer_id"] == item["model_id"] and
                    model["tokenizer_revision"] == item["model_revision"] and
                    model["max_length"] == 128,
                    f"{item['slot']} model/tokenizer/length mismatch")
            if item["slot"] == "xlm_roberta":
                require(model["preprocessing"]["profile_config_sha256"] == PROFILE_SHA and
                        model["pooling"] == "first_token", "XLM-R profile/pooling mismatch")
    else:
        config = read_json(run_dir / "config.json")
        metrics = read_json(run_dir / "metrics.json")
        require(config["model"] == "svm" and config["seed"] == item["seed"] and
                config["preprocessing"] == "manuscript_aligned_cleaning" and
                abs(metrics["val_weighted_f1"] - item["validation_weighted_f1"]) < 1e-12,
                "Historical Linear SVM config/validation mismatch")
    return {"slot": item["slot"], "source_run": item["run"],
            "artifact": str(artifact.relative_to(ROOT)), "sha256": observed_sha,
            "validation_weighted_f1": item["validation_weighted_f1"],
            "source_manifested": item["expected_sha256"] is not None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    status = {"run_id": OUT.name, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(OUT / "run_status.json", status)
    try:
        mask = check_mask()
        artifacts = [check_artifact(item) for item in ROSTER]
        report = {
            "schema_version": "teke-external-inference-artifact-preflight-v1",
            "scientific_status": "pre_inference_no_model_loaded",
            "mask": mask,
            "roster": artifacts,
            "limitations": [
                "This preflight does not load checkpoints or prove runtime model compatibility.",
                "The historical SVM bundle has no source artifact manifest; its hash is captured here, not matched to a training-time hash.",
                "BERTurk's older checkpoint metadata lacks the immutable profile field; canonical cleaning equivalence must be enforced by the inference runner.",
                "No Teke predictions, metrics, fitting, or calibration occurred.",
            ],
        }
        write_json(OUT / "report.json", report)
        write_json(OUT / "manifest.json", {"report.json": sha256(OUT / "report.json")})
        status["status"] = "complete"
        print(f"Preflight complete: {OUT.relative_to(ROOT)}", flush=True)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(OUT / "run_status.json", status)


if __name__ == "__main__":
    main()
