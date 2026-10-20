"""Run the original four-profile corpus audit in a NEW offline execution copy.

Uses cached pinned data and already reconstructed assignments. No model/tokenizer
loading, fitting, inference or publication. All generated files are local only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 1:
        raise ValueError("This bounded offline helper takes no arguments")
    if not (ROOT / "workspace_receipt.json").is_file():
        raise ValueError("Run only in an isolated execution copy")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg")
    sys.path.insert(0, str(ROOT))
    attempts = []

    def block_network(event, arguments):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Network disabled during profile reconstruction")

    sys.addaudithook(block_network)
    from scripts.audit_preprocessing_profiles import main as corpus_audit
    corpus_audit(["--offline"])
    if attempts:
        raise RuntimeError("A network attempt occurred, even if caught by an imported library")
    print("Four-profile reconstruction completed offline; no model/tokenizer loading or fitting.")


if __name__ == "__main__":
    main()
