"""Audit exact pinned-tokenizer lengths and truncation at 128/256/512."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_artifacts import ExperimentRun, repo_relative, sha256_file
from src.preprocessing import (
    fingerprint_preprocessed_splits,
    load_preprocessing_profile,
    load_profile_fingerprint_manifest,
    preprocessing_profile_record,
    verify_split_fingerprints,
)
from src.protocol_constants import (
    EXPERIMENT_SCHEMA_VERSION,
    ID_TO_LABEL,
    WINVOKER_ASSIGNMENTS_PATH,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
)


CONFIG_PATH = ROOT_DIR / "configs" / "audits" / "winvoker_tokenizer_lengths_v1.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "audits" / "winvoker_tokenizer_lengths_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-complete", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "tokenizer-length-audit-config-v1.0":
        raise ValueError("Unsupported tokenizer-length audit config schema")
    if (
        config.get("dataset_id") != WINVOKER_DATASET_ID
        or config.get("dataset_revision") != WINVOKER_DATASET_REVISION
        or config.get("thresholds_including_special_tokens") != [128, 256, 512]
    ):
        raise ValueError("Tokenizer audit config differs from the locked dataset/thresholds")
    tokenization = config.get("tokenization", {})
    if tokenization != {
        "add_special_tokens": True,
        "truncation": False,
        "padding": False,
        "return_length": True,
        "batch_size": 512,
    }:
        raise ValueError("Tokenizer audit must measure untruncated exact lengths")
    if (
        resolve(config.get("split_assignments", ""))
        != WINVOKER_ASSIGNMENTS_PATH.resolve()
        or sha256_file(WINVOKER_ASSIGNMENTS_PATH) != WINVOKER_ASSIGNMENTS_SHA256
    ):
        raise ValueError("Tokenizer audit does not name the locked split assignments")
    if config.get("report_slices") != [
        "overall",
        "split",
        "split_label",
        "split_source",
    ]:
        raise ValueError("Tokenizer audit slices differ from the locked contract")
    quantiles = config.get("quantiles")
    if not isinstance(quantiles, list) or not quantiles or any(
        not isinstance(value, (int, float)) or not 0 < value < 1 for value in quantiles
    ):
        raise ValueError("Tokenizer audit quantiles must be a non-empty list within (0, 1)")
    if not config.get("combinations"):
        raise ValueError("Tokenizer audit requires at least one profile/tokenizer combination")
    return config


def tokenize_lengths(tokenizer, texts: Iterable[str], batch_size: int) -> np.ndarray:
    values = list(texts)
    lengths: list[int] = []
    for start in range(0, len(values), batch_size):
        encoding = tokenizer(
            values[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        batch_lengths = [len(input_ids) for input_ids in encoding["input_ids"]]
        reported_lengths = [int(value) for value in encoding["length"]]
        if batch_lengths != reported_lengths:
            raise ValueError("Tokenizer-reported lengths differ from exact input-ID lengths")
        lengths.extend(batch_lengths)
    result = np.asarray(lengths, dtype=np.int64)
    if len(result) != len(values) or (result <= 0).any():
        raise ValueError("Tokenizer length audit produced missing or non-positive lengths")
    return result


def iter_groups(frame: pd.DataFrame):
    yield "overall", None, None, None, frame
    for split, group in frame.groupby("split", sort=False):
        yield "split", split, None, None, group
    for (split, label), group in frame.groupby(["split", "label"], sort=False):
        yield "split_label", split, label, None, group
    for (split, source), group in frame.groupby(["split", "source"], sort=False):
        yield "split_source", split, None, source, group


def summarize_lengths(
    frame: pd.DataFrame,
    combination: dict[str, str],
    quantiles: list[float],
    thresholds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    length_rows: list[dict[str, Any]] = []
    truncation_rows: list[dict[str, Any]] = []
    for group_type, split, label, source, group in iter_groups(frame):
        values = group["token_length"].to_numpy(dtype=np.int64)
        base = {
            **combination,
            "group_type": group_type,
            "split": split,
            "label": label,
            "source": source,
            "rows": len(group),
        }
        summary = {
            **base,
            "minimum": int(values.min()),
            "mean": float(values.mean()),
            "maximum": int(values.max()),
            "quantile_method": "numpy_linear",
        }
        for quantile in quantiles:
            summary[f"p{int(quantile * 100):02d}"] = float(
                np.quantile(values, quantile, method="linear")
            )
        length_rows.append(summary)
        for threshold in thresholds:
            count = int((values > threshold).sum())
            excess = np.maximum(values - threshold, 0)
            truncation_rows.append(
                {
                    **base,
                    "max_length": threshold,
                    "rows_truncated": count,
                    "truncation_rate": count / len(values),
                    "tokens_removed_total": int(excess.sum()),
                    "tokens_removed_mean_all_rows": float(excess.mean()),
                    "tokens_removed_mean_truncated_rows": (
                        float(excess[excess > 0].mean()) if count else 0.0
                    ),
                }
            )
    return length_rows, truncation_rows


def tokenizer_record(tokenizer, model_config: dict[str, Any]) -> dict[str, Any]:
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    return {
        "model_family": model_config["model_family"],
        "tokenizer_id": model_config["tokenizer_id"],
        "requested_revision": model_config["tokenizer_revision"],
        "reported_commit_hash": init_kwargs.get("_commit_hash"),
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(tokenizer.vocab_size),
        "model_max_length": int(tokenizer.model_max_length),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    config = load_config(config_path)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # These imports initialize Hugging Face offline constants, so they must
    # occur only after the command-line mode has been applied.
    from transformers import AutoTokenizer

    from scripts.run_transformer_experiment import load_model_config
    from src.data_loader import load_winvoker_dataset, preprocess_dataset_with_profile

    combination_specs: list[dict[str, Any]] = []
    profile_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    for group in config["combinations"]:
        profile_path, profile = load_preprocessing_profile(resolve(group["profile"]))
        previous_profile_path = profile_paths.get(profile["profile_id"])
        if previous_profile_path is not None and previous_profile_path != profile_path:
            raise ValueError("One profile ID resolves to multiple profile config files")
        profile_paths[profile["profile_id"]] = profile_path
        for model_path_value in group["model_configs"]:
            model_path, model_config = load_model_config(resolve(model_path_value))
            previous_model_path = model_paths.get(model_config["model_family"])
            if previous_model_path is not None and previous_model_path != model_path:
                raise ValueError("One model family resolves to multiple model config files")
            model_paths[model_config["model_family"]] = model_path
            combination_specs.append(
                {
                    "profile_id": profile["profile_id"],
                    "model_family": model_config["model_family"],
                }
            )
    unique_keys = {(item["profile_id"], item["model_family"]) for item in combination_specs}
    if len(unique_keys) != len(combination_specs):
        raise ValueError("Tokenizer audit contains duplicate profile/tokenizer combinations")

    run_config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": config["run_id"],
        "run_kind": "tokenizer_length_audit",
        "training_or_model_loading": False,
        "tokenizer_loading": True,
        "audit_config": repo_relative(config_path),
        "audit_config_sha256": sha256_file(config_path),
        "offline": args.offline,
        "local_files_only": args.local_files_only,
        "resume_requested": args.resume,
        "combinations": combination_specs,
    }
    with ExperimentRun(
        output_dir,
        run_id=config["run_id"],
        run_kind="tokenizer_length_audit",
        config=run_config,
        runtime_environment={"device": "cpu", "precision_mode": "not_applicable"},
        overwrite_complete=args.overwrite_complete,
        resume_incomplete=args.resume,
    ) as run:
        model_configs = {
            family: load_model_config(path)[1] for family, path in model_paths.items()
        }
        tokenizers = {}
        for family, model_config in model_configs.items():
            print(
                "Checking pinned tokenizer cache: "
                f"{family}={model_config['tokenizer_id']}@"
                f"{model_config['tokenizer_revision']}"
            )
            try:
                tokenizers[family] = AutoTokenizer.from_pretrained(
                    model_config["tokenizer_id"],
                    revision=model_config["tokenizer_revision"],
                    local_files_only=args.local_files_only,
                )
            except Exception as error:
                raise RuntimeError(
                    "Pinned tokenizer is unavailable before corpus processing: "
                    f"{family}={model_config['tokenizer_id']}@"
                    f"{model_config['tokenizer_revision']}"
                ) from error
            print(f"Pinned tokenizer ready: {family}")
        tokenizer_metadata = {
            family: tokenizer_record(tokenizers[family], config_value)
            for family, config_value in model_configs.items()
        }
        for family, metadata in tokenizer_metadata.items():
            reported_commit = metadata["reported_commit_hash"]
            if reported_commit is not None and reported_commit != metadata["requested_revision"]:
                raise ValueError(
                    f"Tokenizer commit mismatch for {family}: "
                    f"{reported_commit} != {metadata['requested_revision']}"
                )

        dataset = load_winvoker_dataset(
            dataset_name=WINVOKER_DATASET_ID,
            revision=WINVOKER_DATASET_REVISION,
        )
        fingerprint_manifest_path = resolve(config["fingerprint_manifest"])
        profiles: dict[str, dict[str, Any]] = {}
        profile_splits: dict[str, dict[str, pd.DataFrame]] = {}
        fingerprint_inputs: dict[str, Any] = {}
        for profile_id, profile_path in profile_paths.items():
            resolved_profile_path, profile = load_preprocessing_profile(profile_path)
            profile_record = preprocessing_profile_record(resolved_profile_path, profile)
            manifest_path, manifest_sha, entry = load_profile_fingerprint_manifest(
                fingerprint_manifest_path,
                profile_record,
            )
            splits = preprocess_dataset_with_profile(
                dataset,
                profile,
                included_splits=("train", "val", "test"),
            )
            actual = fingerprint_preprocessed_splits(splits)
            verify_split_fingerprints(actual, entry["splits"])
            profiles[profile_id] = profile
            profile_splits[profile_id] = splits
            fingerprint_inputs[profile_id] = {
                "manifest": repo_relative(manifest_path),
                "manifest_sha256": manifest_sha,
                "profile_config_sha256": profile_record["profile_config_sha256"],
                "profile_definition_sha256": profile_record["profile_definition_sha256"],
                "verified_split_fingerprints": actual,
            }

        progress_path = run.path("progress.json")
        progress = []
        length_rows = []
        truncation_rows = []
        if args.resume and progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8")).get(
                "completed", []
            )
            length_path = run.path("length_summary.csv")
            truncation_path = run.path("truncation_summary.csv")
            if not length_path.is_file() or not truncation_path.is_file():
                raise ValueError(
                    "Cannot resume tokenizer audit: progress exists without both summary files"
                )
            length_rows = pd.read_csv(length_path).to_dict("records")
            truncation_rows = pd.read_csv(truncation_path).to_dict("records")
        completed_keys = {
            (item["profile_id"], item["model_family"])
            for item in progress
            if item.get("status") == "complete"
        }
        configured_keys = {
            (item["profile_id"], item["model_family"]) for item in combination_specs
        }
        if len(completed_keys) != len(progress) or not completed_keys <= configured_keys:
            raise ValueError("Tokenizer-audit progress is duplicate, malformed, or out of scope")
        for rows, artifact_name in (
            (length_rows, "length_summary.csv"),
            (truncation_rows, "truncation_summary.csv"),
        ):
            artifact_keys = {
                (row["profile_id"], row["model_family"]) for row in rows
            }
            if not completed_keys <= artifact_keys or not artifact_keys <= configured_keys:
                raise ValueError(
                    f"Cannot resume tokenizer audit: {artifact_name} and progress disagree"
                )
            rows[:] = [
                row
                for row in rows
                if (row["profile_id"], row["model_family"]) in completed_keys
            ]
        for combination_index, item in enumerate(combination_specs, start=1):
            profile_id = item["profile_id"]
            family = item["model_family"]
            if (profile_id, family) in completed_keys:
                print(
                    f"Skipping completed tokenizer audit {combination_index}/"
                    f"{len(combination_specs)}: {profile_id} + {family}"
                )
                continue
            print(
                f"Starting tokenizer audit {combination_index}/{len(combination_specs)}: "
                f"{profile_id} + {family}"
            )
            frames = []
            for split_name, frame in profile_splits[profile_id].items():
                lengths = tokenize_lengths(
                    tokenizers[family],
                    frame["text"].astype(str),
                    int(config["tokenization"]["batch_size"]),
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "split": "validation" if split_name == "val" else split_name,
                            "label": frame["label"].map(ID_TO_LABEL).to_numpy(),
                            "source": frame["dataset"].astype(str).to_numpy(),
                            "token_length": lengths,
                        }
                    )
                )
            combined = pd.concat(frames, ignore_index=True)
            identity = {
                "profile_id": profile_id,
                "model_family": family,
                "tokenizer_id": model_configs[family]["tokenizer_id"],
                "tokenizer_revision": model_configs[family]["tokenizer_revision"],
            }
            combo_lengths, combo_truncation = summarize_lengths(
                combined,
                identity,
                list(config["quantiles"]),
                list(config["thresholds_including_special_tokens"]),
            )
            length_rows.extend(combo_lengths)
            truncation_rows.extend(combo_truncation)
            progress.append({**identity, "rows_tokenized": len(combined), "status": "complete"})
            run.write_csv("length_summary.csv", length_rows)
            run.write_csv("truncation_summary.csv", truncation_rows)
            run.write_json("progress.json", {"completed": progress})
            print(
                f"Completed tokenizer audit {combination_index}/{len(combination_specs)}: "
                f"{profile_id} + {family} ({len(combined)} rows)"
            )

        final_completed_keys = {
            (item["profile_id"], item["model_family"])
            for item in progress
            if item.get("status") == "complete"
        }
        if final_completed_keys != configured_keys:
            raise RuntimeError("Tokenizer audit did not finish every configured combination")

        probes = ["[URL]", "https://example.com", "😊", "Harika 😊", "Kötü mü?"]
        probe_rows = []
        for family, tokenizer in tokenizers.items():
            for text in probes:
                encoded = tokenizer(text, add_special_tokens=True, truncation=False)
                probe_rows.append(
                    {
                        "model_family": family,
                        "tokenizer_id": model_configs[family]["tokenizer_id"],
                        "tokenizer_revision": model_configs[family]["tokenizer_revision"],
                        "text": text,
                        "tokens_json": json.dumps(tokenizer.convert_ids_to_tokens(encoded["input_ids"]), ensure_ascii=False),
                        "input_ids_json": json.dumps(encoded["input_ids"]),
                        "token_length": len(encoded["input_ids"]),
                    }
                )
        run.write_json("tokenizer_metadata.json", tokenizer_metadata)
        run.write_json("fingerprint_inputs.json", fingerprint_inputs)
        run.write_csv("tokenizer_probes.csv", probe_rows)
        run.write_json("audit_config_snapshot.json", config)
        for family, model_config in model_configs.items():
            run.write_json(f"model_configs/{family}.json", model_config)
        for profile_id, profile in profiles.items():
            run.write_json(f"profiles/{profile_id}.json", profile)
        run.write_text(
            "notes.md",
            "# Exact Tokenizer Length Audit\n\n"
            "- No model weights, optimizer, inference, or training are used.\n"
            "- Lengths include each tokenizer's special tokens and are measured without truncation.\n"
            "- Truncation means exact token length greater than 128, 256, or 512.\n"
            f"- Completed profile/tokenizer combinations: {len(progress)}.\n",
        )
        required = [
            "progress.json",
            "length_summary.csv",
            "truncation_summary.csv",
            "tokenizer_metadata.json",
            "fingerprint_inputs.json",
            "tokenizer_probes.csv",
            "audit_config_snapshot.json",
            "notes.md",
        ]
        required.extend(f"model_configs/{family}.json" for family in model_configs)
        required.extend(f"profiles/{profile_id}.json" for profile_id in profiles)
        run.declare_complete(required)


if __name__ == "__main__":
    main()
