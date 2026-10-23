# Validation-gated independent per-question edits: full_question_centered

Question: does likelihood-directed prompt editing yield validation-admitted updates and improve frozen-prompt test accuracy?
Reference diagnostic:20261019T140919Z-directional-fixed-base-batch4-probe-gpu01.
Base validation input:20261019T125813Z-directional-boxed-parseable-16k-one-step-gpu01; source was interrupted after its complete300-row baseline (230/300).
Reuse only complete verified baseline records/summary; do not relabel source or reuse its partial test.

User-approved: Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218, BF16, native40960 context,
DAPO seed0 canonical train150/validation300/test300 with the existing boxed-only protocol.
Task max16384, editor max8192, thinking on, T0.6/top-p0.95/top-k20. No request sampling seeds.
Five rounds of four distinct train questions x3 fresh rollouts; common search_seed0 schedule.
Each question independently proposes/refines once under the round parent. Verify exact original
answer-parseable response IDs; show editors only post-thinking text. Invalid task responses are excluded.
All candidates must individually reinforce successes/suppress failures; no epsilon added.

Condition:scope=full, correction=question_mean, score_unit=mean, branch_threshold=None.
Response-only masks positions after closed thinking, retaining original reasoning as conditioning.
Question centering subtracts same-question average delta across retained rollouts and needs both reward classes.
Branch-only fixes positions with parent top1 probability<0.95; excluded positions contribute zero.
Full-vocabulary T1 scoring is distinct from truncated rollout sampling.

Validate each distinct likelihood-eligible candidate on all300 validation questions, one generation each.
A candidate must score>=stored parent accuracy. Pick highest validation accuracy, then its verified
source-question signed mean-LP gain; one adoption maximum per round. Same-string duplicates are
validated once per round, attributed to their first qualifying source in schedule order.
Freeze the final prompt before initial/selected test,300 questions x1 each. Reuse initial test internally
when selected prompt is identical. Test never selects prompts or settings.

Requested changes from last diagnostic: enable actual validation-gated updates and run all4 approved likelihood conditions.
Derived engineering changes: validation_gated_per_example mode, task budget12060 (60train+at most40x300candidate validations;
reused base costs0), per-question malformed-output handling and punctuation-only body rejection.
Semantic generality remains an editor instruction, not a verified property. No retries/resampling of editor outputs.
Two existing GPU servers run independent conditions; concurrency16 per GPU, same serving capacity and model settings.
All scientific/config/source identities are frozen before launch. Finite command batches continue other conditions if one fails;
report failures explicitly; no automatic parameter relaxation or choosing a successful retry.

Report: per-question evidence/reward composition; measured/verified proposal/refinement counts and LP deltas;
validation candidates and adoption chain; base/final validation and frozen test accuracy; branch coverage;
invalid outputs and incomplete/failed executions. Use run artifacts, not runtime task-call estimates, as evidence.
