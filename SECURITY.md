# Safe local use and release checks

Run processing only in a NEW isolated execution copy outside this distributable
folder. Do not execute a historical script merely to inspect it: many entry
points fit models, infer, download inputs or write files when launched. Read the
reproducibility source map and start with preparation-only paths.

- Never commit credentials, tokens, environment files, caches, datasets, model
  files, predictions, scores, logs or generated figures/results.
- Keep authentication tokens in the local environment, not source or shared logs.
- Load only trusted local pickle/checkpoint files. Deserializing untrusted Python
  objects can execute code. New checkpoint reloads refer to the current local
  run; pretrained assets remain separately acquired inputs under their own terms.
- Do not disable immutable input checks, remove private historical gates, relax
  expected hashes or overwrite failed/completed runs just to obtain a pass.
- Public replication preparation is the default. Fitting requires explicit mode
  and confirmation; it is never a side effect of resolving a recipe.

Release checks enumerate allowlisted code/config/instruction files, inspect
UTF-8/JSON/Python syntax and static imports, check original-source copy identities,
look for result/data/binary files and screen selected credential/attribution
patterns. Candidate findings must be reviewed without publishing secret values.
These are scoped checks, not a guarantee that code has no vulnerabilities or a
complete secret scan. No external security service is contacted.

No public issue tracker/contact URL is invented here. Resolve a suspected secret
or unexpected dependency locally before authorizing any external deposit.
