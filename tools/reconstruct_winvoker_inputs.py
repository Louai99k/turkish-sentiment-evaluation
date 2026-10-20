"""Reconstruct locked split membership and exact-text masks without saved results.

Run only in a workspace produced by create_workspace.py, never repository/ or
the original research workspace. Default offline; --allow-download is explicit.
No fit, checkpoint loading, inference, prediction comparison or publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "workspace_receipt.json").is_file():
        raise ValueError("Create a NEW isolated execution workspace first; original artifacts must not be touched")
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        def block_network(event, arguments):
            if event == "socket.connect":
                raise RuntimeError("Network disabled in offline input reconstruction")

        sys.addaudithook(block_network)
    sys.path.insert(0, str(ROOT))
    # Import after offline policy; imported helpers do not run the historical audit main.
    import numpy as np
    from datasets import load_dataset
    from scripts.audit_historic_split_lineage import (
        LABEL_TO_ID, atomic_write_csv, build_historic_split, build_test_mask, split_tables,
    )
    from src.data_loader import clean_text
    from src.protocol_constants import (
        WINVOKER_DATASET_ID, WINVOKER_DATASET_REVISION, WINVOKER_SPLIT_SEED,
        WINVOKER_ASSIGNMENTS_SHA256, WINVOKER_TEST_MASK_SHA256,
    )

    output = ROOT / "outputs/audits/historic_split_lineage_v1"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "running", "error": None, "scientific_fit_or_inference": False}
    status_path = output / "reconstruction_status.json"

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write(status_path, status)
    try:
        dataset = load_dataset(WINVOKER_DATASET_ID, revision=WINVOKER_DATASET_REVISION)
        frame = dataset["train"].select_columns(["text", "label", "dataset"]).to_pandas()
        if len(frame) != 440679:
            raise ValueError("Pinned publisher training population changed")
        frame.insert(0, "source_row_index", np.arange(len(frame), dtype=np.int64))
        frame["__raw_text"] = frame["text"].fillna("").astype(str)
        frame["__clean_text"] = frame["__raw_text"].map(clean_text)
        frame["label_id"] = frame["label"].map(LABEL_TO_ID)
        if frame["label_id"].isna().any():
            raise ValueError("Unknown labels")
        frame["label_id"] = frame["label_id"].astype(int)
        splits = build_historic_split(frame, WINVOKER_SPLIT_SEED)
        _, _, _, assignments = split_tables(splits)
        mask, _, _ = build_test_mask(frame, splits)
        pins = {"split_assignments.csv": (assignments, WINVOKER_ASSIGNMENTS_SHA256),
                "test_inclusion_mask.csv": (mask, WINVOKER_TEST_MASK_SHA256)}
        observations = {}
        for name, (table, expected) in pins.items():
            path = output / name
            atomic_write_csv(path, table)
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            observations[name] = {"expected_sha256": expected, "observed_sha256": observed,
                                  "match": expected == observed}
        write(output / "reconstruction_identity.json", observations)
        if not all(record["match"] for record in observations.values()):
            raise ValueError("Reconstructed bytes differ from locked inputs; stop, do not relax pins")
        status["status"] = "complete"
        print("Reconstructed assignment/mask bytes match locked identities. No existing predictions required.")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write(status_path, status)


if __name__ == "__main__":
    main()
