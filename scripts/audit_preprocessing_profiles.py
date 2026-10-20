"""Audit Winvoker emoji/URL prevalence and immutable preprocessing fingerprints."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import regex

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiment_artifacts import ExperimentRun, repo_relative, sha256_file
from src.preprocessing import (
    FINGERPRINT_SCHEMA_VERSION,
    ORDERED_FINGERPRINT_ALGORITHM,
    fingerprint_preprocessed_splits,
    load_preprocessing_profile,
    preprocessing_profile_record,
)
from src.protocol_constants import (
    EXPERIMENT_SCHEMA_VERSION,
    ID_TO_LABEL,
    WINVOKER_ASSIGNMENTS_PATH,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
)


CONFIG_PATH = ROOT_DIR / "configs" / "audits" / "winvoker_preprocessing_profiles_v2.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "audits" / "winvoker_preprocessing_profiles_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite-complete", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "preprocessing-corpus-audit-config-v1.0":
        raise ValueError("Unsupported preprocessing corpus-audit config schema")
    if (
        config.get("dataset_id") != WINVOKER_DATASET_ID
        or config.get("dataset_revision") != WINVOKER_DATASET_REVISION
    ):
        raise ValueError("Audit config does not name the locked Winvoker snapshot")
    if (
        resolve(config.get("split_assignments", ""))
        != WINVOKER_ASSIGNMENTS_PATH.resolve()
        or sha256_file(WINVOKER_ASSIGNMENTS_PATH) != WINVOKER_ASSIGNMENTS_SHA256
    ):
        raise ValueError("Audit config does not name the locked split-assignment artifact")
    if config.get("prevalence_slices") != [
        "overall",
        "split",
        "split_label",
        "split_source",
    ]:
        raise ValueError("Preprocessing audit slices differ from the locked contract")
    if int(config.get("manual_example_limit_per_feature_split", 0)) < 1:
        raise ValueError("Preprocessing audit requires at least one manual example per slice")
    return config


def compile_patterns(config: dict[str, Any]):
    url_spec = config["feature_patterns"]["url"]
    emoji_spec = config["feature_patterns"]["emoji_sequence"]
    if url_spec["engine"] != "python_re" or emoji_spec["engine"] != "regex_v1":
        raise ValueError("Feature-pattern engines differ from the locked audit contract")
    return {
        "url": re.compile(url_spec["pattern"]),
        "emoji_sequence": regex.compile(emoji_spec["pattern"]),
    }


def feature_vectors(texts: pd.Series, pattern) -> tuple[pd.Series, pd.Series]:
    counts = texts.astype(str).map(lambda value: len(list(pattern.finditer(value))))
    return counts.gt(0), counts


def prevalence_row(
    frame: pd.DataFrame,
    feature: str,
    group_type: str,
    has_feature: pd.Series,
    occurrences: pd.Series,
    split: str | None = None,
    label: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    rows = len(frame)
    present = int(has_feature.sum())
    return {
        "feature": feature,
        "group_type": group_type,
        "split": split,
        "label": label,
        "source": source,
        "rows": rows,
        "rows_with_feature": present,
        "feature_occurrences": int(occurrences.sum()),
        "row_prevalence": present / rows if rows else None,
    }


def prevalence_records(raw: pd.DataFrame, patterns: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature, pattern in patterns.items():
        has_feature, occurrences = feature_vectors(raw["raw_text"], pattern)
        records.append(
            prevalence_row(raw, feature, "overall", has_feature, occurrences)
        )
        for split, indices in raw.groupby("split", sort=False).groups.items():
            group = raw.loc[indices]
            records.append(
                prevalence_row(
                    group,
                    feature,
                    "split",
                    has_feature.loc[indices],
                    occurrences.loc[indices],
                    split=split,
                )
            )
        for (split, label), indices in raw.groupby(["split", "label"], sort=False).groups.items():
            group = raw.loc[indices]
            records.append(
                prevalence_row(
                    group,
                    feature,
                    "split_label",
                    has_feature.loc[indices],
                    occurrences.loc[indices],
                    split=split,
                    label=label,
                )
            )
        for (split, source), indices in raw.groupby(["split", "source"], sort=False).groups.items():
            group = raw.loc[indices]
            records.append(
                prevalence_row(
                    group,
                    feature,
                    "split_source",
                    has_feature.loc[indices],
                    occurrences.loc[indices],
                    split=split,
                    source=source,
                )
            )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    config = load_config(config_path)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Import the Hugging Face datasets stack only after offline mode is fixed.
    from src.data_loader import (
        clean_text,
        load_winvoker_dataset,
        preprocess_dataset_with_profile,
    )

    patterns = compile_patterns(config)
    loaded_profiles = [load_preprocessing_profile(resolve(path)) for path in config["profiles"]]
    records = [preprocessing_profile_record(path, profile) for path, profile in loaded_profiles]
    profiles = {profile["profile_id"]: profile for _, profile in loaded_profiles}
    record_by_id = {record["profile_id"]: record for record in records}
    canonical_id = "winvoker_canonical_v1"
    minimal_id = "winvoker_minimal_strip_v1"
    required_profile_ids = {
        canonical_id,
        minimal_id,
        "winvoker_canonical_emoji_preserve_v2",
        "winvoker_canonical_url_placeholder_v1",
    }
    if set(profiles) != required_profile_ids or len(records) != len(required_profile_ids):
        raise ValueError("Preprocessing audit must contain each locked profile exactly once")

    canonical_url_pattern = next(
        operation["pattern"]
        for operation in profiles[canonical_id]["ordered_transformations"]
        if operation["operation_id"] == "remove_urls"
    )
    emoji_protected_pattern = next(
        operation["protected_pattern"]
        for operation in profiles["winvoker_canonical_emoji_preserve_v2"][
            "ordered_transformations"
        ]
        if operation["type"] == "regex_substitution_preserving_matches"
    )
    if canonical_url_pattern != config["feature_patterns"]["url"]["pattern"]:
        raise ValueError("URL prevalence pattern differs from the canonical profile")
    if emoji_protected_pattern != config["feature_patterns"]["emoji_sequence"]["pattern"]:
        raise ValueError("Emoji prevalence pattern differs from the preservation profile")

    run_config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": config["run_id"],
        "run_kind": "preprocessing_corpus_audit",
        "training_or_model_or_tokenizer_loading": False,
        "audit_config": repo_relative(config_path),
        "audit_config_sha256": sha256_file(config_path),
        "offline": args.offline,
        "profiles": records,
    }
    with ExperimentRun(
        output_dir,
        run_id=config["run_id"],
        run_kind="preprocessing_corpus_audit",
        config=run_config,
        runtime_environment={"device": "cpu", "precision_mode": "not_applicable"},
        overwrite_complete=args.overwrite_complete,
    ) as run:
        dataset = load_winvoker_dataset(
            dataset_name=WINVOKER_DATASET_ID,
            revision=WINVOKER_DATASET_REVISION,
        )
        profile_splits = {
            profile_id: preprocess_dataset_with_profile(
                dataset,
                profile,
                included_splits=("train", "val", "test"),
            )
            for profile_id, profile in profiles.items()
        }
        fingerprints = {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "dataset_id": WINVOKER_DATASET_ID,
            "dataset_revision": WINVOKER_DATASET_REVISION,
            "split_assignments_sha256": WINVOKER_ASSIGNMENTS_SHA256,
            "fingerprint_algorithm": ORDERED_FINGERPRINT_ALGORITHM,
            "profiles": {
                profile_id: {
                    **record_by_id[profile_id],
                    "splits": fingerprint_preprocessed_splits(splits),
                }
                for profile_id, splits in profile_splits.items()
            },
        }

        upstream = pd.DataFrame(dataset["train"])
        raw_frames = []
        for split, frame in profile_splits[minimal_id].items():
            source_rows = frame["source_row_index"].to_numpy(dtype="int64")
            raw_frames.append(
                pd.DataFrame(
                    {
                        "split": "validation" if split == "val" else split,
                        "source_row_index": frame["source_row_index"],
                        "source": frame["dataset"].astype(str),
                        "label": frame["label"].map(ID_TO_LABEL),
                        "raw_text": upstream.iloc[source_rows]["text"].astype(str).to_numpy(),
                    }
                )
            )
        raw = pd.concat(raw_frames, ignore_index=True)
        prevalence = prevalence_records(raw, patterns)

        comparison_rows: list[dict[str, Any]] = []
        equivalence_rows: list[dict[str, Any]] = []
        for split in ("train", "val", "test"):
            canonical = profile_splits[canonical_id][split]["text"].astype(str).reset_index(drop=True)
            legacy = profile_splits[minimal_id][split]["text"].astype(str).map(clean_text).reset_index(drop=True)
            mismatches = int((canonical != legacy).sum())
            equivalence_rows.append(
                {
                    "split": "validation" if split == "val" else split,
                    "rows": len(canonical),
                    "mismatches": mismatches,
                    "exact_equivalence": mismatches == 0,
                }
            )
            for profile_id, splits in profile_splits.items():
                values = splits[split]["text"].astype(str).reset_index(drop=True)
                comparison_rows.append(
                    {
                        "profile_id": profile_id,
                        "split": "validation" if split == "val" else split,
                        "rows": len(values),
                        "rows_changed_vs_canonical": int((values != canonical).sum()),
                        "empty_rows": int(values.eq("").sum()),
                        "total_characters": int(values.str.len().sum()),
                        "rows_with_url_pattern_after_preprocessing": int(
                            values.map(lambda value: bool(patterns["url"].search(value))).sum()
                        ),
                        "rows_with_emoji_after_preprocessing": int(
                            values.map(
                                lambda value: bool(patterns["emoji_sequence"].search(value))
                            ).sum()
                        ),
                    }
                )
        if any(row["mismatches"] for row in equivalence_rows):
            raise RuntimeError("Canonical profile differs from src.data_loader.clean_text")

        factor_profiles = {
            "winvoker_canonical_emoji_preserve_v2": "emoji_sequence",
            "winvoker_canonical_url_placeholder_v1": "url",
        }
        factor_isolation_rows: list[dict[str, Any]] = []
        for split in ("train", "val", "test"):
            reported_split = "validation" if split == "val" else split
            raw_split = raw.loc[raw["split"].eq(reported_split)].reset_index(drop=True)
            canonical_frame = profile_splits[canonical_id][split].reset_index(drop=True)
            for profile_id, feature in factor_profiles.items():
                profile_frame = profile_splits[profile_id][split].reset_index(drop=True)
                if not raw_split["source_row_index"].equals(
                    profile_frame["source_row_index"]
                ) or not canonical_frame["source_row_index"].equals(
                    profile_frame["source_row_index"]
                ):
                    raise RuntimeError(
                        f"Factor-isolation row order mismatch for {profile_id}/{reported_split}"
                    )
                has_factor = raw_split["raw_text"].map(
                    lambda value: bool(patterns[feature].search(value))
                )
                changed = profile_frame["text"].astype(str).ne(
                    canonical_frame["text"].astype(str)
                )
                factor_isolation_rows.append(
                    {
                        "profile_id": profile_id,
                        "factor": feature,
                        "split": reported_split,
                        "rows": len(profile_frame),
                        "rows_with_factor": int(has_factor.sum()),
                        "rows_changed_vs_canonical": int(changed.sum()),
                        "changed_rows_without_factor": int((changed & ~has_factor).sum()),
                        "factor_rows_unchanged": int((has_factor & ~changed).sum()),
                    }
                )
        unexpected_factor_changes = sum(
            row["changed_rows_without_factor"] for row in factor_isolation_rows
        )
        if unexpected_factor_changes:
            raise RuntimeError(
                "A one-factor profile changed rows without its declared raw-text factor: "
                f"{unexpected_factor_changes}"
            )

        profile_lookups = {
            (profile_id, split): frame.set_index("source_row_index")["text"]
            for profile_id, splits in profile_splits.items()
            for split, frame in splits.items()
        }
        example_rows: list[dict[str, Any]] = []
        limit = int(config["manual_example_limit_per_feature_split"])
        for feature, pattern in patterns.items():
            for split in ("train", "validation", "test"):
                matches = raw.loc[
                    raw["split"].eq(split)
                    & raw["raw_text"].map(lambda value: bool(pattern.search(value)))
                ].head(limit)
                for row in matches.itertuples(index=False):
                    source_index = int(row.source_row_index)
                    example = {
                        "feature": feature,
                        "split": split,
                        "source_row_index": source_index,
                        "source": row.source,
                        "label": row.label,
                        "raw_text": row.raw_text,
                    }
                    split_key = "val" if split == "validation" else split
                    for profile_id in profile_splits:
                        example[profile_id] = str(
                            profile_lookups[(profile_id, split_key)].loc[source_index]
                        )
                    example_rows.append(example)

        run.write_json("profile_fingerprints.json", fingerprints)
        run.write_csv("feature_prevalence.csv", prevalence)
        run.write_csv("profile_change_summary.csv", comparison_rows)
        run.write_csv("factor_isolation.csv", factor_isolation_rows)
        run.write_csv("canonical_equivalence.csv", equivalence_rows)
        run.write_csv("manual_examples.csv", example_rows)
        run.write_json("audit_config_snapshot.json", config)
        for profile_id, profile in profiles.items():
            run.write_json(f"profiles/{profile_id}.json", profile)
        overall = [row for row in prevalence if row["group_type"] == "overall"]
        run.write_text(
            "notes.md",
            "# Winvoker Preprocessing/Feature Audit\n\n"
            "- No tokenizer, model, inference, or training is used.\n"
            "- Canonical equivalence must be exact across train/validation/test.\n"
            + "\n".join(
                f"- {row['feature']}: {row['rows_with_feature']}/{row['rows']} rows "
                f"({row['row_prevalence']:.6%})."
                for row in overall
            )
            + "\n- one-factor isolation: 0 changed rows lacked their declared factor."
            + "\n",
        )
        required = [
            "profile_fingerprints.json",
            "feature_prevalence.csv",
            "profile_change_summary.csv",
            "factor_isolation.csv",
            "canonical_equivalence.csv",
            "manual_examples.csv",
            "audit_config_snapshot.json",
            "notes.md",
        ]
        required.extend(f"profiles/{profile_id}.json" for profile_id in profiles)
        run.declare_complete(required)


if __name__ == "__main__":
    main()
