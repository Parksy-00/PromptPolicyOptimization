# PromptPolicyOptimization

Fixed-trajectory likelihood editing, TextGrad optimization and archived likelihood analyses.
Model weights remain frozen. GEPA uses **PromptConsolidation's existing artifact implementation**
at `src/apo/gepa/gepa_artifact/`, through its native runner and separate environment.
The standalone framework GEPA optimizer and recipes were removed; historical artifacts and
source snapshots are preserved. `vendor/gepa/` remains only for historical provenance.

```bash
uv sync --frozen --extra dev --extra local
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

See [code map](docs/codemap.md) for the active editing and analysis extension points.

## Archived TextGrad analysis

The independent [TextGrad likelihood audit](docs/textgrad-likelihood.md) scores fixed archived
train and full optimization-validation responses under their actual parent and proposed prompts.
It includes accepted and rejected proposals, original validation decisions, a separate repeat8
validation comparison, per-token traces and independent verification. It performs no generation
and does not perform optimization.

## Online TextGrad

The existing CLI also supports official TextGrad with numerical fixed-parent success-likelihood
feedback and a verified positive-gain admission filter. See the [method](docs/textgrad-online.md)
and the [DAPO pilot plan](experiments/textgrad_dapo_likelihood_pilot/plan_report.md).
Install the `textgrad` extra alongside the local/dev extras. This is a separate optimizer path;
GEPA runs use the separate artifact implementation described above.


## Likelihood editing with validation

Train rollout -> propose -> fixed-response likelihood verification -> refine -> reverify.
Base validation is evaluated before training; each distinct verified edit gets complete validation.
Require nondecreasing validation accuracy, select by validation then train likelihood gain, or keep
parent. Default5 iterations; freeze the last accepted instruction before final test.

Optional scoring scopes are full trajectory, visible response only, and fixed parent branch positions;
same-question repeated rollouts allow mean-drift correction. Raw likelihood values are always retained.
See [pipeline](docs/contrastive-edit-refinement.md), [code map](docs/codemap.md), and the
[300-question validation comparison plan](experiments/directional_validation_full_raw/plan_report.md).
Local vLLM replicas distribute generation across GPU0 andGPU1 through the existing CLI.

```bash
uv run --no-sync prompt-policy experiments/directional_validation_full_raw/experiment.toml --dry-run
```



### Store generation-time token probabilities

For vLLM rollout sampling probabilities (after temperature, top-k and top-p), start the
server with `--logprobs-mode processed_logprobs`, retain the exact startup command in
its server manifest, and add to the experiment's existing `[serving]` section:

```toml
rollout_logprobs = "processed_logprobs"
```

Rollout records contain `rollout_token_logprobs`, aligned one-to-one with the original
`response_ids` including EOS, plus `rollout_logprobs_mode` and `rollout_sampling`. Their
sum is the generation-time sequence logprob (for length-stopped responses, the generated
prefix only). `raw_logprobs` is available for pre-filter model probabilities and must not
be called the sampling policy likelihood. The server mode must match; otherwise startup
fails. Omitting the option preserves the existing no-capture behavior. This adds no
teacher-forcing request and does not recover probabilities absent from old rollouts.

Capturing the actual parent probability does not guarantee candidate-prompt likelihood
will be nonzero: changed top-k/top-p support can still exclude a fixed original token.
