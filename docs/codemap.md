# Code map

| Responsibility | Extension point |
|---|---|
| CLI, dry-run, provenance, final validation-only selection and test | `src/prompt_policy/cli.py` |
| Standalone TOML validation | `src/prompt_policy/config.py` |
| JSONL split identity and task reward | `src/prompt_policy/benchmarks.py` |
| Local Qwen/HF rendering, generation and model identity | `src/prompt_policy/backends.py::LocalHFBackend` |
| Exact token logprob calculation and immutable-response cache | `src/prompt_policy/scoring.py` |
| Question-balanced objective and accuracy guard | `src/prompt_policy/objective.py` |
| GEPA integration | `src/prompt_policy/optimizer.py` |
| GEPA task/reward/feedback adapter | `TaskAdapter` |
| Local acceptance hook with frozen parent trajectories | `LikelihoodAcceptance` |
| Same-model reflection | `LocalReflection` |
| Actual-rollout/iteration limits | `SearchStopper` |
| Append-only records and cost accounting | `src/prompt_policy/artifacts.py` |

GEPA source is an unmodified pinned snapshot in `vendor/gepa/`, installed through the uv path
source. Its `AcceptanceCriterion` hook supplies the full proposal, before/after captures and
optimizer state. This avoids patching GEPA or replacing its search, reflection, Pareto selection,
validation and native state machinery. Verify the source inventory before every CLI invocation.

Keep task reward, likelihood measurement, local acceptance and final evaluation separate. New
objectives extend `objective.py` and `LikelihoodAcceptance`; they must not alter benchmark reward
or write a relative local gain into GEPA's global scores. A future TextGrad proposal backend can
reuse `TrajectoryScorer`, `Trajectory` and the objective without importing consolidation code.

The serial numerical kernel was adapted from the reference listed in `SOURCE_PROVENANCE.json`.
Unlike that Qwen-specific unmasked right-padding path, it does not batch/pad trajectories. Numerical
tests compare it with full causal-LM logits on a tiny random Qwen3 model and chunk-size changes.
