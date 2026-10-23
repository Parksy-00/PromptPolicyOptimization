# Validation-gated likelihood editing: comparison plan

Status: boxed/generalization implementation verified; revised research runs have not been launched. Prior tag-format execution was interrupted and preserved. Full train150 / validation300 / test300 approved by user; execution authorized. Validation is1 repeat per question by explicit user correction; fixed seed0 training schedule is shared across conditions.

## Question and comparison

Does a likelihood-verified instruction edit maintain or improve actual validation accuracy?
After establishing the validation gate, compare four independent 5-iteration searches:

| Condition | Fixed token mask | Likelihood change used for direction checks |
| --- | --- | --- |
| full_raw | Entire captured response including thinking and sampled EOS | Raw mean log-prob change |
| response_raw | Only positions after the final closed `</think>` | Raw mean log-prob change |
| full_question_centered | Entire captured response | Raw mean change minus the same-question rollout average |
| branch_raw | Positions where parent full-vocabulary max token probability < 0.95 | Raw mean change over selected positions |

These are separate changes from full_raw, not cumulative ablations. All four use 3 rollouts of the same
training question per iteration, so the centering comparison does not also change rollout count.
The user approved question-mean centering with mixed reward classes and the parent max-probability
0.95 branch definition. No claim is made that 80% of this model's tokens are deterministic; the actual
selected fraction will be reported.

## Inputs and identity

Method/source reference: tagged one-step continuation
`20261019T074120Z-directional-edit-tagged-one-step-gpu1-resume`.
It has no validation and used answer_tag_integer_v1. The user subsequently requested the answer rule
in USER, then explicitly replaced answer tags with boxed output and requested generalizable lessons
in proposer/refiner instructions. Current recipes use paired boxed_integer_v1 rendering/scoring:
editable instruction in SYSTEM; immutable boxed rule + `Question:` in USER. Both editors require
reusable lessons for similar problems and prohibit hard-coding observed numbers/equations/objects/answers.
Initial instruction remains `Solve the problem and provide the answer in the correct format.`

Dataset: canonical `common.benchmarks.load_benchmark("DAPOMathBench", 0)` exported through the isolated
GEPA environment. Canonical benchmark fingerprint:
`331f651942894b54de9a183b715bf3c8ce1beacf74b2ed5cb25be3ffa8451914`.
Prepared full split sizes: train150 / validation300 / test300.
The ordered questions AND gold answers exactly match the archived TextGrad seed0 split
`20260917T062343Z-dapo-textgrad-ctx40-bbd2/native/splits.json` (sha256
`d7d9685b2206e588c05e38c400d1b7748ef6adddc6a759d3f1eb3a02512361f8`).
All 300 validation questions also match the ordered questions attached to GEPA seed0 run
`20261005T1222Z-dapo-s0-gepa-qwen3_8b-mc2000-aa18`.
Historical GEPA response captures are reconstructed: only their question identities were used for
this cross-check, not their completions/rewards or their optimizer state.
Canonical sample IDs differ from archived `dapo-source-*` IDs; the explicit mapping is preserved in
`artifacts/inputs/dapo_seed0_canonical_150_300_300/sample_id_map.json`.

There is one whitespace-normalized duplicate QUESTION pair inside canonical test (indices125/295,
both gold80), with distinct canonical IDs and different original bytes. Preserve both original rows
and test300 exactly; no cross-split question overlap is allowed. The explicit within-split duplicate
policy is fingerprinted/reported. No duplicate validation/training questions exist.

This experiment uses the user-requested boxed integer reward, not historical GEPA/DSPy's
parser; historical accuracy is not treated as the new baseline accuracy.

Model: frozen Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, native context40960.
Generation AND reflection: temperature0.6, top-p0.95, top-k20, max8192, thinking enabled.
Teacher forcing is full-vocabulary T1, not the truncated generation policy; exact original IDs retained.
No request-level sampling seed. Search RNG0 controls only the common 5-question schedule.

## Pipeline and admission

1. Generate complete base validation: 300 questions x1 repeat, accuracy averaged across all300.
2. For each of exactly5 outer iterations, sample one train question and generate3 parent rollouts.
3. Proposer sees actual labeled full trajectories; emit only `<instruction>...</instruction>`.
4. Teacher-force original response IDs under parent/proposal using the condition's frozen mask.
5. Refiner sees parent/proposal, actual trajectories, mean log-probs and changes, the declared
   scoring scope, and for centering both raw changes and the measured same-question average.
6. Re-score fixed responses under the refinement. Each observed success must improve and each
   failure decline under the declared criterion. Same-class groups cannot verify centered changes.
   Empty response/branch masks are unmeasurable and fail verification; never silently drop those
   trajectories or assign them probability one. A length-limited unclosed thinking response has no
   post-thinking tokens and therefore cannot verify response_raw.
7. Fresh complete validation for EACH DISTINCT likelihood-verified proposal/refinement.
   Require candidate accuracy >= the stored parent's accuracy (ties allowed). Choose greatest
   validation accuracy, then question-balanced signed train mean gain; exact ties favor proposal.
   Otherwise retain parent. Validation questions/responses never enter editor requests.
8. Stop at5 iterations; store actual completed count/reason. Budget3315 reserves base300 and
   5*(train3 + up to2*validation300), preventing mid-iteration exhaustion.
9. Freeze selected prompt, then initial/selected test300 x1. If unchanged, reuse the initial test.
   Test never selects prompts/thresholds. Repeated adaptive validation is not independent test evidence.

Response-only scores retain the entire original thinking prefix as conditioning context: they measure
p(response | instruction, question, original thinking), not a marginal probability integrating over thinking.
Branch-only scores are a masked diagnostic, not the normalized likelihood of the full trajectory;
omitted positions contribute zero to the sum, and mean is over selected positions.
Centering measures relative reinforcement; a positive centered change need not be absolute ascent
or guarantee task-accuracy improvement. Raw changes remain recorded.
The mean score unit is common across these four new runs; historical minimal-loop ranking used
sequence sums, although its editor also saw mean values.

## Local execution and monitoring

Two independent pinned-model vLLM0.11.1 replicas on physical GPU0/GPU1, loopback8016/8017.
Each full validation/test batch is distributed across BOTH GPUs; conditions execute sequentially:
full_raw, response_raw, full_question_centered, branch_raw.
Primary handles editor and teacher-forcing calls. All replica manifests/model hashes/live versions
are checked. Max32 concurrent generation requests total, max16 active sequences/server,
gpu_memory_utilization0.65, max_num_batched_tokens2048, eager, no prefix cache, generation defaults
explicitly disabled. Operational memory settings preserve the prior OOM fix.
Validation outputs are saved as they finish; progress.json/stdout updates every10 completions.
No scheduler/database/new CLI is added; use the existing prompt-policy entrypoint.

## Requested, derived, unresolved

Requested: add base validation and post-likelihood validation gate; default5 iterations; compare
response-only, repeated-rollout drift correction, branch-token scoring; use local GPU0 andGPU1;
validation300 at actual GEPA scale. Drift/branch definitions were explicitly approved.
Requested in the latest revision: paired boxed renderer/scorer and generalizable editor lessons.
Derived: last visible boxed integer extraction, without other-format fallback; mean-unit comparison; 3 rollouts
per question across all conditions; validation1 repeat; nondecrease gate with validation-first ranking;
fixed masks; explicit unmeasurable cases; two replicas and streamed records.
User decision: retain original-scale train150 and test300. The same fixed seed0 shuffled training schedule is reused across conditions; it is not resampled independently. With minibatch1 and5 iterations this visits5 unique training questions, not the entire150-question pool. Validation1 repeat means300 response scores per prompt. The previous tag-format r1 comparison (base run 20261019T091801Z-directional-validation-r1-full-raw-gpu01) was interrupted for this requested revision; its complete/partial outputs are preserved. New boxed runs must start with fresh base validation, new run IDs and updated source fingerprints; no tag-format baseline reuse. This revision has been dry-run verified; no revised scientific run has started.
Worst-case per condition: search3315 task generations + test600 + editor10. Actual validation cost
is lower when likelihood rejects candidates; no candidates are evaluated on validation just to fill a budget.
These runs share data/schedule/configuration but draw fresh stochastic responses, not paired identical
rollouts. One search seed and5 iterations are a mechanism/selection comparison, not a robust method ranking.

## Outputs

Per-run config/manifest/source hashes and snapshot; validation.jsonl + validation_scores.jsonl;
full original train trajectories/optimizer calls; token log-probs (branch also top1); fixed token masks;
raw/centered per-trajectory changes; edits/decisions; candidate parent chain; frozen selection; final test.
Comparison report: base/selected validation and delta, validation increases/ties/rejections, likelihood
pass rate split by success/failure composition, accepted updates, raw versus relative gains, mask coverage,
format/length failures, frozen test delta, actual token/call/wall costs. Include prompt traces for actual
accepted/rejected edits, with no claim of effectiveness from likelihood alone.

## Current editor evidence revision

The current implementation uses answer-parseable training trajectories only for editing and likelihood
verification. Editors see only post-thinking response text; the retained original full token IDs and
configured likelihood masks are unchanged. Wrong but parseable answers remain failures. Empty pools
consume one skipped iteration without retry/resampling or editor/scoring/candidate-validation calls.
All raw captures and exclusion identities are preserved. Validation/test still score all rows. This
revision has not been launched as a new research run; previous executions keep their source snapshots.
