# Working in this repository

- Do not invoke Superpowers unless the user explicitly requests it.
- This is an independent project. Do not import or mutate PromptConsolidation, `common`, or `mopcd`.
- Read `docs/codemap.md` before extending the method; update it when extension points change.
- Keep `vendor/gepa/` unchanged. Its revision and SHA inventory are pinned in `UPSTREAM.json`.
- Preserve exact generated response token IDs. Candidate likelihood must score the parent's
  fixed trajectories. Keep global validation scores as accuracy, not parent-relative gain.
- Keep train/validation/test separate. Never use test to choose prompts or objective weights.
- Scientific parameters belong in standalone TOML files; no YAML/TOML experiment inheritance.
- Do not add request-level sampling seeds. Search RNG and statistical RNG are separate.
- Use `uv`; run tests and lint in this repository's own environment.
- Implementation tests and dry-runs are authorized by coding requests. Launch a real research
  run only after the user approves its explicit conditions and data/model identities.
- Store each scientific execution under `artifacts/runs/<run-id>/`; do not commit research outputs,
  add schedulers/databases, delete remote data or upload results automatically.
