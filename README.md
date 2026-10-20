# Turkish sentiment evaluation — code-only package

Project source, configurations and instructions for the reported Turkish
sentiment-analysis setups: TF-IDF classifiers, training-vocabulary fastText +
Logistic Regression, BiLSTM, BERTurk and Transformer comparators. Original
historical workflow code and independently identified replication paths are
distinguished; this package contains no experimental outputs.

## Start here

1. Read [environment requirements](docs/ENVIRONMENT.md) and
   [reproducibility/input reconstruction](docs/REPRODUCIBILITY.md).
2. Use `tools/create_workspace.py` to create a NEW local execution copy outside
   this package. Keep all generated data, logs and model files there.
3. Reconstruct the locked assignment/mask and preprocessing inputs using the
   documented helpers; acquire upstream data/model/vector inputs separately.
4. Read [replication recipes and execution safeguards](docs/REPRODUCTION_RECIPES.md).
   `scripts/run_reproduction.py` defaults to preparation only. Fitting requires
   separate explicit confirmation and unchanged prior source/input preparation.
   No automatic seed sweep, download or restart is supported.

`src/` contains reusable scientific infrastructure; `scripts/` contains original
and replication workflows; `configs/` contains non-result identity/setup
definitions; `tools/` contains bounded checks and isolated-workspace helpers.
Some historical scripts deliberately require private original artifacts/gates.
Their inclusion is not a claim they are turnkey fresh-clone commands; do not
remove those safeguards or fabricate their original approval records.

## Verification scope

Author-executed isolated checks passed for twelve selected module imports,
synthetic preprocessing/metrics, exact split/mask reconstruction, all four profile
identities/twelve split fingerprints and the complete fingerprint document.
All thirteen replication setups resolved in both stages; twenty-four real
prepare-only CLI invocations passed, with zero fitting or network attempts.
fastText configuration resolution and missing-archive rejection were checked;
the full vector archive was not loaded by these preparation checks.

Actual replication fitting/inference and a clean dependency installation have
NOT been verified. Reusing recorded helpers is not numerical-parity proof.
Historical library-default gaps remain disclosed; no missing seed variability or
original training records are manufactured. Keep these limitations visible.

## Contents and exclusions

Code/configurations/instructions only. No datasets, split CSVs, generated masks,
fingerprint files, scientific score tables/statistics, predictions/logits, model
weights, vectors, figures, PDFs or other generated research assets. Article and
supplement are separate. See [EXCLUSIONS.md](EXCLUSIONS.md). Code may contain
fixed input identities and historical invariant checks; that does not distribute
the referenced private artifacts or make those gates portable.

## License and safe use

Original software Luay Kelani owns and may license uses
**PolyForm Noncommercial License1.0.0**, with the Required Notice in [NOTICE](NOTICE).
Read [LICENSE.md](LICENSE.md), [LICENSING.md](LICENSING.md),
[THIRD_PARTY.md](THIRD_PARTY.md) and [SECURITY.md](SECURITY.md) for exact scope,
retained third-party rights and safe local use. Dataset/model terms are separate;
this is not unrestricted permissive open source or a blanket ownership claim.

## Repository, citation and archive status

Code repository: https://github.com/Louai99k/turkish-sentiment-evaluation .
Software version: **1.0.0**. See [CITATION.cff](CITATION.cff) for citation metadata.
Zenodo DOI **10.5281/zenodo.23509979** was reserved by the author for this software
release. At preparation of this snapshot the Zenodo record is still an unpublished
draft; DOI reservation is not evidence that the archive is publicly available.
The archive will be accessible at https://doi.org/10.5281/zenodo.23509979 after
publication. No article DOI or publication date is inferred from this identifier.

Luay Kelani developed the software. Ramin Abbaszadi provided supervision and
methodological guidance; this software citation does not change article authorship.
Luay is performing the code-only release himself. The documented verification
limits above remain unchanged; release does not certify unexecuted model fits.
See [RELEASE_REQUIREMENTS.md](RELEASE_REQUIREMENTS.md) for release boundaries.
