# TextGrad with verifiable success-likelihood ascent

The official TextGrad operations are used unchanged from revision
`75e912e210864b61999781778cdf756d4468120f`, the same revision as the archived DAPO
study: `Variable`, `LLMCall.backward`, `StringBasedFunction`, `sum`, and
`TextualGradientDescent`. The dependency and source inventory are checked before execution.
See https://github.com/zou-group/textgrad/tree/75e912e210864b61999781778cdf756d4468120f.
This module is a new forward/measurement/acceptance integration, not an exact replay of the
historical loop or a replacement numerical autograd engine.

For each outer iteration, generate a train minibatch under the admitted parent prompt p.
Preserve all generated IDs, sampled EOS and binary task rewards. Freeze this reference pool
for every inner proposal. The objective is

```
G(p', p) = mean_questions mean_rollouts r(y) [sum_tokens log pi_p'(y|x) - sum_tokens log pi_p(y|x)].
```

Failed rollouts contribute zero and stay in each question denominator. Unlike the existing
GEPA objective, this is a reward-weighted sequence statistic, not a success-conditioned
average of token means. Teacher forcing uses the same response IDs under both prompts.
The first candidate is optimized from task-accuracy feedback plus a numerical StringBasedFunction
loss `-G` on the parent success pool. Each following inner candidate uses freshly captured task
responses under the preceding proposal and the measured `-G` relative to the same original
parent. Candidate task responses never replace parent success references.

Upstream textual backward interprets these numeric loss outputs; TGD proposes text edits.
There is no parameter gradient computation or claim of an exact differentiable policy gradient.
Measurable ascent is enforced outside the LLM: eligibility requires G > tolerance and observed
validation accuracy >= parent's recorded validation accuracy (with configured numerical tolerance).
This enforces average ascent on the fixed train pool; individual successes may still decrease.
A no-success train pool is skipped because ascent is unverified. Among eligible inner proposals,
choose the highest G, breaking ties with validation accuracy. If none pass, keep the parent.
Only then draw a new parent reference pool for the next outer iteration.

The control uses the same pinned TextGrad graph and number of inner proposals, but hides the
likelihood loss from backward and uses accuracy eligibility/selection. Measurements remain
available for diagnostic comparison. It is a matched proposal-budget control, not an exact
reproduction of historical one-proposal-per-batch TextGrad. Joint feedback+gate effects are
not separated by this first comparison. Actual usage can differ after a no-success skip.

Final selection considers only admitted prompts, using validation accuracy, with later admitted
prompts breaking ties. The selection is written before initial/selected prompt test evaluation.
A likelihood-rejected prompt cannot return through global validation selection. Validation guards
are measurements on the optimization split; they do not guarantee true accuracy improvement.

The pilot uses full-vocabulary T=1 task sampling (top-p=1, top-k=0) and T=1 scoring, to avoid the
historical filtered-sampling mismatch. Sampling remains stochastic, without per-request seeds.
The finite token cap defines truncated response prefixes; length stops are recorded. Teacher
forcing does not invent EOS. Current pilot data and settings are declared in
[the plan](../experiments/textgrad_dapo_likelihood_pilot/plan_report.md).

Install the isolated optional dependency and inspect both standalone configs:

```bash
uv sync --frozen --extra dev --extra local --extra analysis --extra textgrad
TEXTGRAD_DISABLE_LOGGING=1 uv run --no-sync prompt-policy experiments/textgrad_dapo_likelihood_pilot/accuracy_control.toml --dry-run
TEXTGRAD_DISABLE_LOGGING=1 uv run --no-sync prompt-policy experiments/textgrad_dapo_likelihood_pilot/likelihood_ascent.toml --dry-run
```

The environment variable disables upstream's extra cwd log; the project records complete
optimizer requests/responses/IDs/phases in `textgrad_calls.jsonl`, textual gradients in
`textgrad_gradients.jsonl`, numerical loss in `textgrad_losses.jsonl`, immutable reference
pools in `textgrad_pools.jsonl`, and measured proposals in `textgrad_proposals.jsonl`.
`decisions.jsonl` records actual admission/restore, `likelihood.jsonl` the exact token scores,
and existing CLI files retain model/source/data identity, selection, held-out results and costs.
Real runs require the explicit condition/data/model approval described in AGENTS.md.
