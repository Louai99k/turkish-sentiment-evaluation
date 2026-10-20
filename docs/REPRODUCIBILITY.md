# What this code-only package can reproduce

This is source/configuration/instruction staging, not a bundle of the paper's
results. No scores, predictions, datasets, generated split masks, figures or
checkpoints are distributed. Code inclusion alone does not prove that a fresh
environment can replay an entire historical workflow. Read the distinctions below.

## Source map

| Workflow | Entry points | Required local inputs |
| --- | --- | --- |
| Data identities and splits | `tools/reconstruct_winvoker_inputs.py`, `scripts/audit_historic_split_lineage.py` | Pinned upstream Winvoker snapshot; no historical predictions needed by the new reconstruction helper. |
| Profiles and token lengths | `audit_preprocessing_profiles.py`, `audit_tokenizer_lengths.py` | Reconstructed assignments, profile definitions, pinned upstream tokenizers. |
| Classical baselines | `run_baseline_experiment.py` | Pinned data; historical runner retains its original behavior and output schema. |
| BERTurk / Transformer comparisons | `run_berturk_experiment.py`, `run_transformer_experiment.py` | Data, model/tokenizer revisions; protocol-v2 paths additionally require reconstructed masks, profile fingerprints and appropriate selection gates. |
| BiLSTM | `run_bilstm_protocol_v2.py` | Data, assignments/masks, profile fingerprints; shared historical neural helper is included as an import dependency, not endorsed as the corrected fastText entry point. |
| Corrected fastText singleton | `run_fasttext_train_vocab_refit.py` | Training-only vocabulary, local vector archive, original parameter/provenance inventory and historical comparison inputs. Private-history replay, NOT fresh-clone turnkey. |
| Complete-record SVM / BERTurk replacements | `run_svm_complete_records.py`, `run_berturk_provenance_replacement.py` | Original historical bundles and accepted private gates. Private-history replay, NOT fresh-clone turnkey. |
| Aggregation and conditional statistics | `aggregate_*`, `significance_tests.py`, `run_berturk_pinned_comparisons.py` | Independently generated or authorized local run artifacts, exact schemas and comparison configurations. |
| Frozen external and constrained polarity | `run_teke_direct_transfer.py`, `extract_*_scores.py`, `calculate_*_polarity_metrics.py` | Lawfully acquired Teke inputs, frozen checkpoints/roster/masks and local score artifacts. No substitute filtering of hard predictions. |
| Attention/embeddings/projections/ROC | `extract_berturk_qualitative.py`, `project_berturk_tfidf.py`, `analyze_berturk_roc_pr.py` | Selected checkpoint, stable sample lineage, local extraction/projection/score artifacts and analysis configuration. |

Names in the table without a directory prefix live under `scripts/`. Local static
import closure is collected by the private staging command; dynamic imports and
constructed file paths still need runtime/manual review. Missing literal file
dependencies are reported, not silently removed or replaced by fabricated gates.

## From a fresh clone

1. Prepare an isolated execution copy using `tools/create_workspace.py` with
   `--destination` naming a NEW directory outside the cloned/staged package.
   Never run experiments inside `repository/`. Install dependencies separately;
   see `ENVIRONMENT.md`. Keep logs and credentials outside the distributable tree.
2. Acquire the Winvoker dataset from Hugging Face using the exact dataset ID and
   revision in `configs/datasets/winvoker_historic_v2.json`. Read its upstream
   terms; the project's software license does not license dataset redistribution.
   The analyzed population is the publisher's training partition, not the union
   of its partitions. No account/token is included.
3. In the NEW workspace run `tools/reconstruct_winvoker_inputs.py`. It defaults
   to offline cached inputs; `--allow-download` explicitly permits upstream access.
   It reuses the original split/mask functions without running their historical
   prediction audit. It reconstructs assignment/mask CSVs and must match both
   locked byte hashes. Failure blocks use; never edit the expected hashes merely
   to get a pass. These local generated CSVs must not enter a public release.
4. Run `tools/reconstruct_profile_fingerprints.py` in the isolated workspace to
   reconstruct all four profiles using the packaged original corpus-audit code.
   It is offline and refuses network connects. The fingerprint document itself
   has no timestamp fields, so both semantic identities and exact bytes can be
   checked against the authors' private reference without distributing that file.
   Other audit status/manifests include runtime metadata; do not fabricate original
   acceptance records or confuse reconstructed inputs with historical approval.
5. Acquire model/tokenizer snapshots from their upstream repositories at the
   configuration's immutable revisions, under their own licenses. Do not load
   arbitrary pickle/checkpoint files from untrusted parties. fastText requires
   a separately acquired local Turkish vector archive; train-only vocabulary and
   singleton lbfgs interpretation must be retained. No vector redistribution is
   authorized here.
6. Choose a genuinely fresh experiment ID and document its complete configuration
   before execution. Historical revision-specific runners intentionally fail closed
   without original acceptance artifacts. They are source evidence of the actual
   process, not universal fresh-clone entry points. A portable workflow must not
   weaken their gates or overwrite historical IDs to pretend it is the same run.

## Configuration boundary

Distributed configurations contain dataset identities, model/tokenizer pins,
preprocessing transformations and non-result corpus/tokenizer audit definitions.
The historical preprocessing freeze includes measured screening outcomes; it is
deliberately NOT distributed under package A. Likewise scientific-result matrices,
provenance acceptance records and private manuscript configurations stay outside.
No result-bearing JSON is smuggled into the package by calling it a configuration.
References to those paths in unchanged source do not distribute their contents.

Current staging therefore is NOT an end-to-end turnkey reproduction of every
paper result. Remaining work is tracked in `RELEASE_REQUIREMENTS.md`: bounded
fresh-package checks, dependency reconstruction/portable execution coverage and
per-file rights review. No new historical-model reruns are required to prepare
the code. Existing scientific results remain accepted independently of this work.

Completed bounded checkpoint: 12 selected local imports, synthetic profile/
fingerprint/metric checks, and independently reconstructed assignment/mask bytes
passed in an isolated execution copy. No training/inference or clean dependency
installation was performed. Four-profile reconstruction subsequently passed too:
all twelve split fingerprints and whole manifest bytes match. Generated audit
outputs stay outside this package. Declarative accepted-setup recipes are in
`configs/reproduction/accepted_setups_v1.json`; portable preparation/explicit fit
dispatch is implemented with bounded user checks pending. See
`REPRODUCTION_RECIPES.md`. No full scientific fit/numerical parity claim is made.

## Scientific interpretation that must remain intact

- fastText + lbfgs Logistic Regression is one deterministic configuration, not a
  five-seed model; embedded historical seed metadata is provenance only.
- Fit vocabulary/vectorizer on training only; checkpoint selection uses recorded
  validation Weighted F1 with the applicable eligibility restriction. Do not
  introduce a Macro F1 selection rule or test-based model selection.
- Screening is validation-only; it must not construct a test loader or emit test
  predictions/metrics. Test-derived descriptive analyses do not select checkpoints.
- Original weighted-CE confirmatory statistics, standard-CE reviewer sensitivity
  and post-review pinned analyses are distinct; nonsignificance is not equivalence.
- External results use the frozen checkpoints and evaluation masks. New internal
  provenance fits do not retrospectively change the external model roster.

Current repository and reserved software DOI are listed in ../README.md. These
reconstruction instructions do not establish publication of the Zenodo draft or
numerical reproduction of unexecuted fits.
