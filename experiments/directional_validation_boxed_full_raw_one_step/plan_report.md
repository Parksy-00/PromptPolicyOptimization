# Boxed one-iteration update check

Approved request: rerun the revised implementation for exactly one iteration to inspect update behavior.
Base/source reference: `20261019T091801Z-directional-validation-r1-full-raw-gpu01` (interrupted, preserved).
The new run uses fresh outputs; no answer-tag validation replay or old rollout reuse.

Requested changes since that run: boxed_integer_v1 output/scoring, generalizable proposer/refiner
lessons without copying problem-specific numbers/equations/objects/answers, no ID beside Example x,
Gold answer label without optimizer-only text, and max_iterations=1 instead of5.

Unchanged scientific settings: Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218;
BF16, thinking enabled, context40960, task and editor T0.6/top-p0.95/top-k20/max8192;
canonical DAPO split seed0 train150/validation300/test300, duplicate test rows preserved;
minibatch1 question x3 rollouts, fixed search seed0 schedule, validation1 draw/question;
full-response mean logprob, no drift centering; each observed success increases and failure decreases;
nondecreasing validation admission (ties allowed); max_task_rollouts3315 remains an upper bound.
Exactly one training question is visited. No other likelihood comparison conditions will run.

Pipeline: fresh base validation300 -> train rollout3 -> propose -> fixed-ID likelihood measurement
-> refine -> remeasure -> each distinct likelihood-verified candidate gets full validation300
-> accept by validation then signed likelihood gain or retain parent -> freeze selection
-> existing CLI evaluates initial/selected prompts on test300 each (reuse initial if unchanged).
Test never selects a prompt. Worst-case search903 generations; final test at most600; editor2.

Derived operational changes: fresh server manifests/processes on GPU0/GPU1, fresh run identity,
and fresh source snapshot. Server arguments otherwise match the approved prior replicas.
Historical scores are not reused as the new baseline. Full response IDs remain fixed likelihood targets.

## Current editor evidence revision

The current implementation uses answer-parseable training trajectories only for editing and likelihood
verification. Editors see only post-thinking response text; the retained original full token IDs and
configured likelihood masks are unchanged. Wrong but parseable answers remain failures. Empty pools
consume one skipped iteration without retry/resampling or editor/scoring/candidate-validation calls.
All raw captures and exclusion identities are preserved. Validation/test still score all rows. This
revision has not been launched as a new research run; previous executions keep their source snapshots.
