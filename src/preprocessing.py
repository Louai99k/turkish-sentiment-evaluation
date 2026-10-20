"""Immutable preprocessing profiles and ordered cleaned-text fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import regex

from src.protocol_constants import (
    ROOT_DIR,
    WINVOKER_ASSIGNMENTS_SHA256,
    WINVOKER_DATASET_ID,
    WINVOKER_DATASET_REVISION,
)


PROFILE_SCHEMA_VERSION = "preprocessing-profile-v1.0"
FINGERPRINT_SCHEMA_VERSION = "preprocessing-fingerprints-v1.0"
ORDERED_FINGERPRINT_ALGORITHM = (
    "sha256(domain='ordered-cleaned-text-v1\\0'; "
    "records=uint64_be_utf8_length||utf8_bytes)"
)
PROFILE_CONFIG_DIR = ROOT_DIR / "configs" / "preprocessing"
REQUIRED_BEHAVIOR_FIELDS = {
    "html",
    "url",
    "mention",
    "punctuation",
    "emoji",
    "whitespace",
    "casing",
}
SUPPORTED_OPERATIONS = {
    "coerce_to_string",
    "strip",
    "regex_substitution",
    "regex_substitution_preserving_matches",
    "collapse_whitespace",
}
OPERATION_ALLOWED_FIELDS = {
    "coerce_to_string": {"order", "operation_id", "type"},
    "strip": {"order", "operation_id", "type", "characters"},
    "regex_substitution": {
        "order",
        "operation_id",
        "type",
        "engine",
        "pattern",
        "replacement",
        "flags",
    },
    "regex_substitution_preserving_matches": {
        "order",
        "operation_id",
        "type",
        "engine",
        "protected_engine",
        "pattern",
        "protected_pattern",
        "replacement",
        "flags",
    },
    "collapse_whitespace": {"order", "operation_id", "type", "replacement"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_profile_path(path: Path | str) -> Path:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    resolved = resolved.resolve()
    try:
        resolved.relative_to(PROFILE_CONFIG_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            f"Preprocessing profile must be inside {PROFILE_CONFIG_DIR.relative_to(ROOT_DIR)}"
        ) from error
    return resolved


def validate_preprocessing_profile(profile: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "profile_id",
        "description",
        "behavior",
        "ordered_transformations",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Preprocessing profile is missing fields: {missing}")
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported preprocessing schema: {profile['schema_version']!r}")
    profile_id = profile["profile_id"]
    if not isinstance(profile_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]*", profile_id
    ):
        raise ValueError(f"Invalid preprocessing profile_id: {profile_id!r}")
    behavior = profile["behavior"]
    if not isinstance(behavior, Mapping) or set(behavior) != REQUIRED_BEHAVIOR_FIELDS:
        raise ValueError(
            "Preprocessing behavior must declare exactly: "
            f"{sorted(REQUIRED_BEHAVIOR_FIELDS)}"
        )
    operations = profile["ordered_transformations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("ordered_transformations must be a non-empty list")
    expected_order = list(range(1, len(operations) + 1))
    actual_order = [operation.get("order") for operation in operations]
    if actual_order != expected_order:
        raise ValueError(
            f"Transformation order must be contiguous and explicit: {actual_order!r}"
        )
    operation_ids: list[str] = []
    for operation in operations:
        operation_id = operation.get("operation_id")
        operation_type = operation.get("type")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("Every transformation requires a non-empty operation_id")
        if operation_type not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported preprocessing operation: {operation_type!r}")
        unknown_fields = set(operation) - OPERATION_ALLOWED_FIELDS[operation_type]
        if unknown_fields:
            raise ValueError(
                f"Unknown fields for {operation_id}/{operation_type}: "
                f"{sorted(unknown_fields)}"
            )
        operation_ids.append(operation_id)
        if operation_type == "regex_substitution":
            if (
                operation.get("engine") != "python_re"
                or "pattern" not in operation
                or "replacement" not in operation
            ):
                raise ValueError(f"Invalid regex_substitution operation: {operation_id}")
            re.compile(
                operation["pattern"],
                flags=_regex_flags(operation.get("flags"), re),
            )
        if operation_type == "regex_substitution_preserving_matches":
            target_engine_name = operation.get("engine")
            protected_engine_name = operation.get(
                "protected_engine", target_engine_name
            )
            if (
                target_engine_name not in {"python_re", "regex_v1"}
                or protected_engine_name != "regex_v1"
                or "pattern" not in operation
                or "protected_pattern" not in operation
                or "replacement" not in operation
            ):
                raise ValueError(
                    f"Invalid regex_substitution_preserving_matches operation: {operation_id}"
                )
            target_engine = re if target_engine_name == "python_re" else regex
            target_flags = _regex_flags(operation.get("flags"), target_engine)
            protected_flags = _regex_flags(operation.get("flags"), regex)
            target_engine.compile(operation["pattern"], flags=target_flags)
            regex.compile(operation["protected_pattern"], flags=protected_flags)
    if len(set(operation_ids)) != len(operation_ids):
        raise ValueError("Preprocessing operation_id values must be unique")


def load_preprocessing_profile(path: Path | str) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_profile_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Preprocessing profile not found: {resolved}")
    profile = json.loads(resolved.read_text(encoding="utf-8"))
    validate_preprocessing_profile(profile)
    return resolved, profile


def preprocessing_profile_record(
    path: Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": profile["schema_version"],
        "profile_id": profile["profile_id"],
        "profile_path": path.resolve().relative_to(ROOT_DIR.resolve()).as_posix(),
        "profile_config_sha256": sha256_file(path),
        "profile_definition_sha256": canonical_json_sha256(profile),
        "behavior": dict(profile["behavior"]),
        "ordered_transformations": list(profile["ordered_transformations"]),
    }


def _regex_flags(names: Sequence[str] | None, engine) -> int:
    value = 0
    for name in names or []:
        if not hasattr(engine, name):
            raise ValueError(f"Unknown regex flag {name!r} for engine {engine.__name__}")
        value |= int(getattr(engine, name))
    return value


def _substitute_preserving_matches(value: str, operation: Mapping[str, Any]) -> str:
    target_engine = re if operation["engine"] == "python_re" else regex
    target_flags = _regex_flags(operation.get("flags"), target_engine)
    protected_flags = _regex_flags(operation.get("flags"), regex)
    protected = regex.compile(operation["protected_pattern"], flags=protected_flags)
    target = target_engine.compile(operation["pattern"], flags=target_flags)
    protected_spans = [match.span() for match in protected.finditer(value)]
    replacement = operation["replacement"]

    def substitute_target(match) -> str:
        start, end = match.span()
        overlaps = [
            (max(start, protected_start), min(end, protected_end))
            for protected_start, protected_end in protected_spans
            if protected_start < end and protected_end > start
        ]
        if not overlaps:
            return replacement
        parts: list[str] = []
        cursor = start
        for protected_start, protected_end in overlaps:
            if cursor < protected_start:
                parts.append(replacement)
            parts.append(value[protected_start:protected_end])
            cursor = protected_end
        if cursor < end:
            parts.append(replacement)
        return "".join(parts)

    return target.sub(substitute_target, value)


def apply_preprocessing_profile(text: Any, profile: Mapping[str, Any]) -> str:
    """Apply the exact ordered transformations declared by ``profile``."""

    value: Any = text
    for operation in profile["ordered_transformations"]:
        operation_type = operation["type"]
        if operation_type == "coerce_to_string":
            value = str(value)
        elif operation_type == "strip":
            value = value.strip(operation.get("characters"))
        elif operation_type == "regex_substitution":
            value = re.sub(
                operation["pattern"],
                operation.get("replacement", ""),
                value,
                flags=_regex_flags(operation.get("flags"), re),
            )
        elif operation_type == "regex_substitution_preserving_matches":
            value = _substitute_preserving_matches(value, operation)
        elif operation_type == "collapse_whitespace":
            value = re.sub(r"\s+", operation.get("replacement", " "), value)
        else:  # Protected by validation; retained as a defensive runtime boundary.
            raise ValueError(f"Unsupported preprocessing operation: {operation_type!r}")
    if not isinstance(value, str):
        raise TypeError("Preprocessing profile did not produce text")
    return value


def ordered_cleaned_text_sha256(texts: Iterable[Any]) -> str:
    """Hash an ordered string sequence using unambiguous length-prefixed UTF-8 records."""

    digest = hashlib.sha256()
    digest.update(b"ordered-cleaned-text-v1\0")
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def fingerprint_preprocessed_splits(
    data_splits: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        ("validation" if split == "val" else split): {
            "rows": int(len(frame)),
            "ordered_cleaned_text_sha256": ordered_cleaned_text_sha256(frame["text"]),
        }
        for split, frame in data_splits.items()
    }


def load_profile_fingerprint_manifest(
    path: Path | str,
    profile_record: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Preprocessing fingerprint manifest is required before screening/final runs: "
            f"{resolved}"
        )
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        raise ValueError("Unsupported preprocessing fingerprint manifest schema")
    if manifest.get("fingerprint_algorithm") != ORDERED_FINGERPRINT_ALGORITHM:
        raise ValueError("Unsupported ordered cleaned-text fingerprint algorithm")
    if (
        manifest.get("dataset_id") != WINVOKER_DATASET_ID
        or manifest.get("dataset_revision") != WINVOKER_DATASET_REVISION
        or manifest.get("split_assignments_sha256") != WINVOKER_ASSIGNMENTS_SHA256
    ):
        raise ValueError("Preprocessing fingerprint manifest dataset/split identity mismatch")
    profile_id = profile_record["profile_id"]
    entry = manifest.get("profiles", {}).get(profile_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Fingerprint manifest has no entry for profile {profile_id!r}")
    identity_fields = (
        "schema_version",
        "profile_id",
        "profile_path",
        "profile_config_sha256",
        "profile_definition_sha256",
        "behavior",
        "ordered_transformations",
    )
    for field in identity_fields:
        if entry.get(field) != profile_record[field]:
            raise ValueError(f"Fingerprint profile {field} mismatch for {profile_id}")
    split_records = entry.get("splits", {})
    if set(split_records) != {"train", "validation", "test"}:
        raise ValueError("Fingerprint entry must contain train, validation, and test")
    for split, record in split_records.items():
        if (
            not isinstance(record.get("rows"), int)
            or record["rows"] <= 0
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("ordered_cleaned_text_sha256", ""))
            )
        ):
            raise ValueError(f"Invalid fingerprint record for {profile_id}/{split}")
    return resolved, sha256_file(resolved), dict(entry)


def verify_split_fingerprints(
    actual: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if not actual:
        raise ValueError("No cleaned-text split fingerprints were computed")
    for split, actual_record in actual.items():
        expected_record = expected.get(split)
        if expected_record is None or dict(actual_record) != dict(expected_record):
            raise ValueError(
                f"Cleaned-text fingerprint mismatch for {split}: "
                f"actual={actual_record}, expected={expected_record}"
            )
