# Validation-gated likelihood editing

The current implementation extends the existing candidate-first loop; no TextGrad/GEPA dependency
or extra diagnosis/hypothesis/merger stage is added. Completed historical runs retain their source snapshots.

```text
Base P -> verified cached validation (if configured), otherwise fresh validation -> stored parent accuracy
  -> repeat 5 times by default:
       train minibatch -> archive all fixed parent rollout IDs/rewards
       -> keep answer-parseable trajectories (correct and incorrect); empty pool: skip iteration
       -> render only post-thinking response text for both editors
       -> proposer -> full instruction C -> likelihood verification
       -> refiner sees actual measured changes -> full instruction R -> likelihood verification
       -> each distinct verified C/R -> complete validation
       -> validation >= parent: select by validation, then signed likelihood gain
       -> otherwise keep parent
  -> freeze selected P -> initial/selected test
```

`search.base_validation_run` optionally points to a prior complete run with the same baseline.
The reused value is checked against model file identity, backend/version, context, generation/repeat
settings, benchmark identity, initial instruction, all actual prompt IDs, recomputed rewards and the
complete source summary. Reused records and their source-file hashes are saved explicitly, with no
new generation cost. Any mismatch fails; it never silently falls back to generation. The editor policy
may differ because it does not affect task baseline rendering/scoring. Source runs remain unchanged.

Validation responses never enter editor prompts. Ties with parent validation are allowed; strictly better
validation takes priority over train likelihood gain. Store rejected validation candidates as well as
accepted ones. Parent validation is reused until adoption; candidate evaluations are fresh. Repeated
adaptive validation can overfit, so frozen endpoint test is separate.

`search.max_iterations` defaults to5 and `validation_rollouts` to1 for this optimizer. Conditions can
explicitly request other values. Both train and validation generations count toward max_task_rollouts;
reserve a complete train iteration plus two complete candidate validations before starting a step.
Search result records iterations_completed and termination_reason; test is outside this budget.

## Likelihood conditions

Optional `[edit_refinement]` fields: likelihood_scope=full|response|branch (default full),
score_unit=sequence|mean (default sequence for older recipes), drift_correction=none|question_mean
(default none). Branch requires an explicit branch_probability_threshold in(0,1).
Question centering requires repeated rollouts and mean units.

Full: all original response IDs, including thinking and actual EOS. Response: only positions after
closing `</think>`, keeping original thinking in the teacher-forcing context. Unclosed thinking has
no visible-response mask. Branch: positions where the PARENT's full-vocabulary top1 probability is
below the threshold; mask fixed across candidates. Other positions contribute zero; report selected
fraction and mean over selected tokens. This masked statistic is not normalized sequence probability.
An empty mask cannot verify any direction and is never silently dropped.

Question mean: delta_i is candidate-minus-parent mean LP; d_x is its average over same-question
fixed rollouts; corrected delta_i=delta_i-d_x. Raw values are retained. Success must have positive
corrected change; failure negative. Homogeneous reward groups are uninformative for this criterion
and fail verification. A relative increase is not an absolute increase or proof of better accuracy.

All modes retain individual direction checks, equal-question averaging, the exact fixed response IDs,
and T1 full-vocabulary scoring distinct from the generation sampling distribution.

## Prompts and task formatting

Proposer sees labeled post-thinking responses and gold answers for answer-parseable training
trajectories, without likelihood values. Parseability uses the same benchmark parser as reward and
is independent of correctness: well-formed wrong answers remain failure evidence. Unclosed
thinking, absent or malformed task answers, and empty visible responses are omitted. A valid response
without a thinking block remains visible. Refiner sees the same retained responses, original/candidate instructions, parent mean LP, measured mean changes,
when enabled, per-question drift and centered changes. Scoring scope stays in configuration and
measurement artifacts; it is not included in the refiner message. The retained trajectories'
complete original IDs remain the likelihood targets; removing displayed thinking does not change
likelihood_scope or turn full-trajectory scoring into response-only scoring. Response-specific feedback is generated from measured changes: increases of failures and decreases
of successes are stated explicitly as undesirable; beneficial changes are identified for preservation.
Semantic interpretation is generated by the refiner. Both must return exactly
one `<instruction>...</instruction>` block. Both request general lessons applicable to other similar problems as reusable reasoning/verification
strategies. This is an editor instruction, not a semantic parsing guarantee.
Parse only its body outside completed thinking; invalid or
length-limited editor output is recorded and fails without fallback/retry.

All raw training captures remain in trajectories.jsonl, including editor_eligible. The separate
editor_evidence.jsonl records inclusion/exclusion identities and raw/retained pool hashes. If no
response is answer-parseable, retain the parent and record a skipped decision with reason
`no parseable training responses`. The batch counts toward max_iterations and task generation
budget; there are no editor/scoring/extra-validation calls, retries or automatic resampling. Validation
and test continue to score every question, including format failures. Historical executions retain
their exact old source snapshots and are not reinterpreted under this filter.

Task SYSTEM contains only editable P. USER begins with the immutable rule:
`Put your final answer at the end of your response in \boxed{...}, with only a plain integer inside (e.g. \boxed{42}).`
Then `Question:` and the original problem. `boxed_integer_v1` scores only the last balanced
`\boxed{...}` outside completed thinking, containing a signed ASCII integer. Multiple intermediate
boxes are allowed, but a missing, unfinished or noninteger final box fails with no Answer/tag/plain-number
fallback. Reward extraction does not crop likelihood targets. Historical protocols retain their
original rendering and scoring. The previous tag-format comparison was interrupted and preserved;
changed-format runs must generate a fresh base validation and use new execution identities.

## Execution and records

VLLMBackend can attach local `[serving].replicas` of the same pinned model. Generation jobs are
round-robin distributed while results retain input order. The primary serves editor and teacher forcing.
Each server's model identity, manifest, dtype/context and live version are checked. Validation records
are appended at completion, with progress every10 jobs. Teacher forcing requests actual-token LP;
branch mode also requests the top1 vocabulary LP at the same causal position.

See [comparison plan](../experiments/directional_validation_full_raw/plan_report.md) for data/model
identity and scientific conditions. New source/configuration does not mutate completed artifacts.


## Editor message roles and layout

Proposer and refiner each receive one USER message and no SYSTEM message.
Proposer begins with "Your task is to write a new instruction for the assistant.", then
`Current Instruction`, `Input`/`Gold answer`, successful/failed responses and editing instructions.
Refiner begins with its own instruction to compare the original and new instructions and refine the
new one. Adjacent `Original Instruction` / `New Instruction` sections precede the input/responses.
Parent mean LP (labeled under the original instruction), new-minus-original mean-LP change and,
when enabled, question drift/corrected change appear immediately below each corresponding response.
`Feedback` follows all responses and identifies each measured increase/decrease and response number,
with an explicit request to lower failed-response likelihood or raise successful-response likelihood
when the opposite happened. Unchanged values and unavailable measurements are described separately.
Centered feedback uses drift-relative mean LP, never mislabeling it as an absolute likelihood change.
There is no generic pass/fail verdict or repeated change-definition sentence. The final paragraph
asks for the refined instruction.
There is no separate scoring-scope line or repeated likelihood-change section. Only mean LP units
are exposed. Measurement masks, direction checks and original fixed response IDs are unchanged.
Output-tag instructions occur once. `optimizer_calls.jsonl` stores exact user `messages`, complete
`request` and `system_prompt=null`. Task messages retain their original task formatting and roles.
