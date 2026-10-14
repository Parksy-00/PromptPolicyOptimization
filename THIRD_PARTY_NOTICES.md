# Third-party source and provenance

`vendor/gepa/` contains the official GEPA Python source, project metadata, README and MIT license
from https://github.com/gepa-ai/gepa at commit
`f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819` (package version 0.1.4).
`UPSTREAM.json` inventories the included files with SHA-256. No upstream source patches were made.
Optional upstream adapters/examples are included for source completeness but are not used by the
new project, and their optional dependencies are not installed unless explicitly requested.

The causal alignment method in `prompt_policy.scoring.token_logprobs` was adapted from
PromptConsolidation's `teacher_prompt_distributions/score.py::response_hidden`.
`SOURCE_PROVENANCE.json` records the exact source bytes and last committed change. No module or
artifact from that project is imported. The existing GEPA-artifact copyright/license notice is
preserved in `LICENSES/gepa-artifact-MIT.txt`. An overall license for the user's new research code
has not been assigned automatically.
