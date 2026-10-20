# Environment and execution boundaries

Reference runtime: Python 3.10.12, PyTorch 2.6.0+cu124, CUDA 12.4 for the
post-review pinned BERTurk fits. `requirements-observed.txt` transcribes recorded
direct dependency versions; SciPy 1.15.3 is additionally recorded in CPU-refit
configuration. It is not a transitive environment lock or a statement that every
historical fit used this environment. tqdm4.70.0 was observed in the 2026-10-20
bounded package check, not retrospectively assigned to all historical fits.
GPU driver/hardware and CPU numerical libraries can affect execution. Hardware
identity and per-run package records belong in locally generated artifacts.

Install Python/PyTorch and these dependencies in a separate environment or Docker
workspace, not by copying the authors' cached environment. Installation and import
checks must be performed by the researcher. Selected module imports and synthetic
checks passed in a fresh execution copy using the existing installed environment.
Exact split/mask reconstruction passed too. A clean dependency installation and
complete execution coverage remain separate gates; no such full claim is made.

Sources import the namespace packages `src` and `scripts`. Run from the isolated
workspace root so those directories resolve locally. Original runner path bootstrap
is preserved; do not set PYTHONPATH to the private research repository to make a
fresh-package check pass.

Never execute a runner just to inspect it: many scripts train, infer, download or
write outputs when launched. The accepted staging audit parses source only. The
prepared `tools/check_workspace.py` imports 12 selected modules and exercises
only synthetic preprocessing/fingerprint/metric functions. It blocks socket
connects and model/tokenizer/checkpoint loaders, hides GPUs and invokes no
training, inference or historical audit main(). The author executed this bounded
check successfully; subsequent edits still require their own inventory/checks.

All generated outputs/caches/logs belong in an isolated local execution workspace,
outside the distributable code package. `tools/create_workspace.py` copies code to
a NEW destination and refuses existing destinations. Its receipt is local only.
Run `tools/reconstruct_winvoker_inputs.py` there, never in the public staging or
original research directory. Model training is a separate explicit operation,
not part of release preparation.
