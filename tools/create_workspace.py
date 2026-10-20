"""Copy code-only sources to a NEW local execution directory outside this package.

No installation, scientific execution, archive creation or network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "scripts", "configs", "docs", "tools"}
TOP_FILES = {".gitignore", "README.md", "LICENSE.md", "NOTICE", "LICENSING.md",
             "EXCLUSIONS.md", "RELEASE_REQUIREMENTS.md", "THIRD_PARTY.md", "SECURITY.md",
             "release_manifest.json", "requirements-observed.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists() or target.is_relative_to(ROOT) or ROOT.is_relative_to(target):
        raise ValueError("Choose a NEW execution directory outside the code package, not its ancestor")
    sources = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed: {relative}")
        if not path.is_file():
            continue
        if len(relative.parts) == 1:
            allowed = relative.as_posix() in TOP_FILES
        else:
            allowed = relative.parts[0] in DIRECTORIES and path.suffix in {".py", ".json", ".md", ".txt"}
        if not allowed:
            raise ValueError(f"Unexpected file; review package boundary: {relative}")
        sources.append(path)
    if not (ROOT / "scripts/run_transformer_experiment.py").is_file():
        raise ValueError("Source staging incomplete; do not create an initial-six-files workspace")
    target.mkdir(parents=True, exist_ok=False)
    inventory = []
    for path in sorted(sources):
        relative = path.relative_to(ROOT)
        payload = path.read_bytes()
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Copy identity failure: {relative}")
        inventory.append({"path": relative.as_posix(), "sha256": digest, "bytes": len(payload)})
    receipt = {"status": "code_workspace_created_no_scientific_execution", "files": inventory,
               "source": str(ROOT), "destination": str(target), "publication": False}
    (target / "workspace_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created isolated execution workspace: {target}")
    print("No dependencies installed, data downloaded, fitting or inference performed.")


if __name__ == "__main__":
    main()
