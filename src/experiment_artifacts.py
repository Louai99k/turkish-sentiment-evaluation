"""Transactional artifact writer for protocol-v2 commands and experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.protocol_constants import EXPERIMENT_SCHEMA_VERSION, ROOT_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n")


def atomic_write_csv(path: Path, rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def environment_payload(runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    distributions = (
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "accelerate",
        "regex",
    )
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes(),
        "packages": {name: package_version(name) for name in distributions},
        "git_commit": git_commit(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime": dict(runtime or {}),
    }


def normalized_invocation(argv: Sequence[str] | None = None) -> str:
    values = list(argv if argv is not None else [sys.executable, *sys.argv])
    return shlex.join(str(value) for value in values)


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_status(output_dir: Path | str) -> dict[str, Any] | None:
    path = Path(output_dir) / "run_status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def complete_run_exists(
    output_dir: Path | str,
    required_files: Iterable[str] = (),
) -> bool:
    output_dir = Path(output_dir)
    status = read_status(output_dir)
    if not status or status.get("status") != "complete":
        return False
    return all((output_dir / relative).is_file() for relative in required_files)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """Expose the standard text-stream capability used by dependencies."""
        primary = self.streams[0] if self.streams else None
        try:
            return bool(primary is not None and primary.isatty())
        except (AttributeError, OSError):
            return False


class ExperimentRun:
    """Manage status, logging, atomic files, failure evidence, and manifest.

    A successful caller must invoke ``declare_complete`` inside the context.
    Exiting normally without that declaration is recorded as a failed run.
    """

    def __init__(
        self,
        output_dir: Path | str,
        *,
        run_id: str,
        run_kind: str,
        config: Mapping[str, Any],
        runtime_environment: Mapping[str, Any] | None = None,
        overwrite_complete: bool = False,
        resume_incomplete: bool = False,
        argv: Sequence[str] | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.run_id = run_id
        self.run_kind = run_kind
        self.config = dict(config)
        self.runtime_environment = dict(runtime_environment or {})
        self.overwrite_complete = overwrite_complete
        self.resume_incomplete = resume_incomplete
        self.invocation = normalized_invocation(argv)
        self.started_at_utc: str | None = None
        self._started_clock: float | None = None
        self._required_files: list[str] = []
        self._completion_declared = False
        self._log_handle = None
        self._stdout_redirect = None
        self._stderr_redirect = None

    def path(self, relative_path: str) -> Path:
        candidate = (self.output_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.output_dir)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes run directory: {relative_path}") from exc
        return candidate

    def write_text(self, relative_path: str, value: str) -> None:
        atomic_write_text(self.path(relative_path), value)

    def write_json(self, relative_path: str, payload: Any) -> None:
        atomic_write_json(self.path(relative_path), payload)

    def write_csv(
        self,
        relative_path: str,
        rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> None:
        atomic_write_csv(self.path(relative_path), rows)

    def _status_payload(
        self,
        status: str,
        *,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        elapsed = None
        if self._started_clock is not None:
            elapsed = time.monotonic() - self._started_clock
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": utc_now() if status != "running" else None,
            "elapsed_seconds": elapsed,
            "required_files": self._required_files,
            "normalized_invocation": self.invocation,
            "error": dict(error) if error else None,
        }

    def __enter__(self) -> "ExperimentRun":
        existing = read_status(self.output_dir)
        directory_is_nonempty = self.output_dir.is_dir() and any(self.output_dir.iterdir())
        if directory_is_nonempty and not self.overwrite_complete:
            existing_status = existing.get("status") if existing else None
            resumable_statuses = {"running", "failed", "interrupted"}
            if not (self.resume_incomplete and existing_status in resumable_statuses):
                if existing_status == "complete":
                    reason = "complete run"
                elif existing_status in resumable_statuses:
                    reason = f"{existing_status} run without the resume flag"
                else:
                    reason = "nonempty unrecognized or legacy run directory"
                raise FileExistsError(
                    f"Refusing to overwrite {reason} {self.output_dir}; "
                    "use the explicit overwrite flag only when replacement is intended"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_utc = utc_now()
        self._started_clock = time.monotonic()
        atomic_write_text(self.path("command.txt"), self.invocation + "\n")
        atomic_write_json(self.path("config.json"), self.config)
        atomic_write_json(
            self.path("environment.json"),
            environment_payload(self.runtime_environment),
        )
        atomic_write_json(self.path("run_status.json"), self._status_payload("running"))

        self._log_handle = self.path("stdout.log").open("a", encoding="utf-8")
        self._log_handle.write(f"\n[{self.started_at_utc}] RUN START {self.run_id}\n")
        self._log_handle.flush()
        tee = Tee(sys.stdout, self._log_handle)
        self._stdout_redirect = redirect_stdout(tee)
        self._stderr_redirect = redirect_stderr(tee)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        print(f"Run ID: {self.run_id}")
        print(f"Output directory: {repo_relative(self.output_dir)}")
        return self

    def declare_complete(self, required_files: Iterable[str]) -> None:
        required = list(dict.fromkeys(str(value) for value in required_files))
        automatically_managed = {
            "run_status.json",
            "config.json",
            "environment.json",
            "command.txt",
            "stdout.log",
            "artifact_manifest.json",
        }
        required = list(dict.fromkeys([*required, *sorted(automatically_managed)]))
        missing = [
            value
            for value in required
            if value != "artifact_manifest.json" and not self.path(value).is_file()
        ]
        if missing:
            raise ValueError(f"Cannot complete run; required artifacts are missing: {missing}")
        self._required_files = required
        self._completion_declared = True

    def _manifest(self, status: str) -> dict[str, Any]:
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if (
                not path.is_file()
                or path.name == "artifact_manifest.json"
                or path.name.endswith(".tmp")
            ):
                continue
            files.append(
                {
                    "path": repo_relative(path),
                    "relative_to_run": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "run_id": self.run_id,
            "run_status": status,
            "manifest_excludes": ["artifact_manifest.json"],
            "files": files,
        }

    def __exit__(self, exception_type, exception, exception_traceback) -> bool:
        missing_completion = exception is None and not self._completion_declared
        if exception is not None:
            traceback.print_exception(exception_type, exception, exception_traceback)
            status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
            error = {
                "type": exception_type.__name__ if exception_type else type(exception).__name__,
                "message": str(exception),
            }
            atomic_write_text(
                self.path("traceback.txt"),
                "".join(traceback.format_exception(exception_type, exception, exception_traceback)),
            )
        elif missing_completion:
            status = "failed"
            error = {
                "type": "CompletionNotDeclared",
                "message": "Run context exited without declare_complete()",
            }
            print(error["message"])
        else:
            status = "complete"
            error = None
            # A resumed run may be completing an output directory that contains
            # the traceback from an earlier failed attempt.  That failure record
            # must not be carried into, or checksummed as part of, the successful
            # artifact bundle.
            self.path("traceback.txt").unlink(missing_ok=True)
            print(f"Run completed: {self.run_id}")

        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(None, None, None)
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(None, None, None)
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()

        atomic_write_json(
            self.path("run_status.json"),
            self._status_payload(status, error=error),
        )
        atomic_write_json(self.path("artifact_manifest.json"), self._manifest(status))

        if missing_completion:
            raise RuntimeError(error["message"])
        return False
