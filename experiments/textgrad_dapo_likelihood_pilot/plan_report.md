# DAPO TextGrad numerical likelihood feedback + ascent filter: pilot plan

Status: implementation and dry-run prepared; real GPU execution awaits approval of these
explicit conditions. User chose numerical feedback plus a likelihood-ascent filter and a small
pilot comparison using the existing DAPO data.

## Question and comparison

Can TextGrad propose and admit prompts that raise the likelihood of fixed parent-success
trajectories, and how does its measured path compare with accuracy-only TextGrad?
Two standalone conditions differ only in `objective.mode` and `likelihood_feedback`:

| Condition | Objective | Numeric likelihood feedback | Positive G gate | Admission ranking |
|---|---|---|---|---|
| accuracy_control | accuracy | false | false | validation accuracy |
| likelihood_ascent | success_likelihood_ascent | true | G > 1e-6 nats/question | largest G, then validation accuracy |

Both conditions measure G and enforce no observed validation-accuracy drop. Both have identical
upstream TextGrad operations and two inner proposals per outer iteration. For the ascent arm,
zero parent successes skip the outer iteration; no candidate passing both guards retains the
parent. The final validation-only choice is restricted to admitted prompts.

## Inputs and immutable identities

Base historical run: `20260917T062343Z-dapo-textgrad-ctx40-bbd2` (seed0).
This starts a new optimization from that run's base instruction; it does not resume its native
optimizer state or reuse old generated trajectories as on-policy samples.

Source split file:
`/home/parksy1202/workspace/PromptConsolidation/artifacts/runs/20260917T062343Z-dapo-textgrad-ctx40-bbd2/native/splits.json`.
Inputs are first6 examples of each archived train/validation/test list, in existing order,
without inspecting rewards to select questions. No overlap across these exported splits.
Identity, selected IDs and hashes are in
`artifacts/inputs/textgrad_dapo_likelihood_pilot_seed0/manifest.json`.
All1,851 archived score/message pairs reproduce with the new historical DSPy renderer/scorer.

Initial instruction: `Solve the problem and provide the answer in the correct format.`
Model for task, backward and TGD: Qwen3-8B, local snapshot revision
`b968826d9c46dd6066d109eabc6255188de91218`.
Official TextGrad revision: `75e912e210864b61999781778cdf756d4468120f`.
Execution: physical local GPU1 only, BF16, context40960, head chunk256, serial inference.

## Explicit scientific settings and changes from the historical run

- Train/validation/test:6/6/6 questions, for a feasibility pilot; historical source sizes are
  recorded in its splits file and are not silently relabeled as the pilot.
- Two outer iterations; train minibatch3; two sequential inner TextGrad proposals each.
- One fresh task response per question in train/validation/test; search RNG0 for minibatches.
  No model sampling seeds are added. New generations in each arm are stochastic and independent.
- Task sampling T=1, top-p=1, top-k=0. This changes historical T=.6/top-p=.95/top-k20 so task
  generation and the teacher-forced scoring policy both use the same full-support distribution.
- Task output cap8192 tokens instead of historical16384, to limit pilot cost. Length stops
  remain observable and may affect rewards. Optimizer sampling T=.6/top-p=.95/top-k=0, output
  cap8192, Qwen thinking enabled as for tasks. This also differs from the historical optimizer
  sampler and cap; both new arms share it.
- Original DSPy field formatting and Python-int scorer are preserved, not the generic numeric
  example scorer. Local Hugging Face inference replaces the historical vLLM API backend.
- Success likelihood uses sequence logprob sums with original EOS IDs, full vocabulary atT=1.
  Primary G retains failures as zero contributions over all sampled train questions/rollouts.
  Token-mean G and success-increase counts are diagnostics. No additive accuracy/likelihood
  weight is introduced; weight0 is explicit because this mode ranks G after the validation guard.
- Max42 search task generations per arm (base validation6 +2×[train3 +2×validation6 +inner train3]).
  At most12 additional test generations per arm, fewer if initial=selected is reused.
  At most28 control /32 treatment optimizer calls, excluded from task budget but fully charged.
  Teacher-forcing costs are also recorded; a token cap is not a runtime guarantee.

The number of outer/inner iterations, reduced splits and caps are implementation choices for the
requested small pilot. Sampling-policy matching and the historical renderer/scorer are derived
choices with the reasons above. These are all listed for approval before execution.

## Measurements and outputs

Each condition creates its own new `artifacts/runs/` directory with full input/model/source
hashes. Record every proposal's fixed-pool identity, G and token-mean G, success-increase count,
validation accuracy, eligibility, actual admission, no-success skips, selected prompt, initial and
selected test accuracy, generated/teacher-forced tokens, optimizer calls and elapsed cost.
The comparison table will show base→selected validation/test scores, accepted updates, all G
values and sign counts per arm. Test is never used in backward, admission or prompt selection.
With6 held-out questions, this measures pipeline feasibility, not general performance improvement.
The two-arm design does not separate the feedback effect from the filter effect.

## Execution after approval

Run sequentially, with no separate service, runner or scheduler:

```bash
CUDA_VISIBLE_DEVICES=1 TEXTGRAD_DISABLE_LOGGING=1 uv run --no-sync prompt-policy experiments/textgrad_dapo_likelihood_pilot/accuracy_control.toml --output artifacts/runs/20261016-textgrad-dapo-pilot-accuracy-gpu1
CUDA_VISIBLE_DEVICES=1 TEXTGRAD_DISABLE_LOGGING=1 uv run --no-sync prompt-policy experiments/textgrad_dapo_likelihood_pilot/likelihood_ascent.toml --output artifacts/runs/20261016-textgrad-dapo-pilot-likelihood-ascent-gpu1
```

No GPU generation has been launched by preparing this plan, tests or dry-runs. Existing outputs
and source DAPO runs remain untouched. Both explicit configs pass the existing CLI dry-run.
