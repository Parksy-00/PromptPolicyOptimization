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

Archived TextGrad likelihood audits use `prompt_policy.analysis.textgrad_likelihood`.
The standalone TOML declares source run IDs, the original validation selection and repeat8
evaluation identities, exact model revision, reconstruction rules and GPU. The reader validates
original rewards and parent/proposal/acceptance links, then reuses the same causal hidden-state
alignment as the scorer. It performs no generation and never imports another project.

Completed audits are independently verified by `prompt_policy.analysis.textgrad_report`, which
checks token caches, original/repeated validation attribution and input hashes before writing a
cross-run report. See [TextGrad likelihood analysis](textgrad-likelihood.md) for the statistic,
policy-gradient interpretation, sampling limitations and commands.

Base-to-final selected endpoint comparisons use `prompt_policy.analysis.textgrad_endpoints`.
They fix original base-validation responses, verify and copy exact cached contexts, and score
missing endpoint contexts with the existing numerical kernel. Endpoint identity is checked
against each native summary and the actual accepted parent chain.

Historical DSPy DAPO endpoints use the same endpoint module with explicit
`task_format=dspy_chat_adapter_v1`. `build_legacy_inputs` reads actual captured endpoint messages,
checks identical wrapper/question history, and verifies all saved rewards via `legacy_dapo_score`.
A cache is optional; the no-cache path hashes the pinned model files and scores all contexts.

Historical candidate-origin comparisons extend the same endpoint reader through explicit
`response_origins` and `include_origin_context`. `make_origin_report` separates final-minus-base
from final-minus-origin, preserves accepted/rejected branch identity, and reports each origin
without pooling. Exact cached endpoint score files can be reused after ID/hash checks.
