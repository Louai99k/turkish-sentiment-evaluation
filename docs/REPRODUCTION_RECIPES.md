# Fixed setups versus original private approval records

`configs/reproduction/accepted_setups_v1.json` contains only executable-setup
parameters and interpretation rules: thirteen reported model/loss configurations.
It contains no experimental scores, predictions, bootstrap outputs, datasets or
models. It is a new code-package recipe, not the original preprocessing freeze,
historical experiment configuration or a fabricated successful validation record.

## Current implementation status

Portable resolution/preparation and explicit fit dispatch are implemented in
`src/reproduction.py` and `scripts/run_reproduction.py`; user checks are pending.
Do not pass the recipe to a historical runner or interpret inclusion as permission
to launch training. Original revision-specific runners retain their original
private gates; none is removed or faked to make these new paths execute.

The new runner requires a fresh path-safe `repro_...` ID and an isolated workspace.
It defaults to preparation, which only resolves parameters and hashes required
inputs/sources. For example, inside the configured Python environment at the
isolated workspace root (PowerShell):

```powershell
python scripts/run_reproduction.py `
  --setup berturk_standard_ce `
  --seed 42 `
  --run-id repro_berturk_standard_ce_check42 `
  --stage validation_only `
  --mode prepare
```

This does not load datasets/models or train. A later, separately requested fit
requires BOTH `--mode fit` and `--confirm-new-fit`, the same arguments and an
unchanged prepared configuration/source/input snapshot. fastText preparation also
requires the separately acquired local archive via `--fasttext-archive` and its
recorded content identity. Singleton setups reject `--seed`; stochastic setups
require one of the fixed seeds. No automatic sweep or download is supported.

`validation_only` never constructs held-out text/loader/predictions/metrics or
reads the test mask. `replication` permits those views after fitting and validation
checkpoint selection, not to tune reported settings. Both stages emit validation
records. Neural checkpoints preserve backbone/pooling/tokenizer/vocabulary and
stable sample lineage; classical bundles record actual effective parameters.

Existing experiment IDs are never overwritten, resumed or restarted by this
runner. It retains epoch recovery files but exposes no automatic resume CLI;
review an interruption before any recovery work. Source hashes are the code
identity authority; a Git commit found in a parent checkout is not proof that
the isolated copy is that commit. No full model-fit numerical parity or clean
dependency installation has been tested at this checkpoint. Preparation checks
explicitly forbid entering fit branches; this limitation must stay visible.

Reported data inputs can now be independently reconstructed: assignments, strict
mask and the four-profile fingerprint document matched original bytes in author-
executed isolated-workspace checks. Those generated files remain local, excluded
from this package. This is not proof of complete model-fitting reproducibility.

## Scientific boundaries

- Transformer/BiLSTM numerical settings come from the recorded comparison/refit
  contracts. Use each model config's actual pooling, immutable checkpoint/tokenizer
  revision, and validation Weighted F1 selection with earliest exact ties.
- No new checkpoint eligibility rule is inferred. Frozen historical external
  representatives/masks are not replaced by a fresh internally fitted model.
- Classical TF-IDF LR/NB/RF entries preserve source constructor arguments, not an
  invented complete inventory of historical library defaults. Resolve and record
  effective parameters and actual package versions for every new replication.
- SVM uses the accepted complete-record refit's explicit classifier arguments.
  It is not a claim that original missing training records were recovered.
- fastText + lbfgs LR is a deterministic singleton. Training-only vocabulary,
  exact/lowercase vector lookup and mean-vector pooling are mandatory. A recorded
  random_state is implementation metadata, not a seed sweep. Deprecated sklearn
  arguments require explicit compatibility handling, not guessed silent changes.
- Five training seeds belong only to the locked stochastic configurations; they
  are not an automatic instruction to rerun any accepted model.
- Attention, embedding projection and saved-logit curve analysis remain descriptive
  post-selection workflows with stable sample lineage. Do not use them to choose
  models against external or held-out test outcomes.

No public-release approval, training authorization, repository URL or Zenodo DOI
is implied. These are independently identified replications, never replacements
for the accepted paper evidence or frozen external model set.
