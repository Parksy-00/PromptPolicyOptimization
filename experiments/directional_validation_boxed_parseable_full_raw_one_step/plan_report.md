# Parseable post-thinking evidence: one-iteration rerun

User approved rerunning the revised implementation under the preceding one-step conditions.
Base run: 20261019T103742Z-directional-boxed-full-raw-one-step-gpu01.
New run uses fresh stochastic generations; no validation replay or rollout reuse.

Requested implementation change: retain only training trajectories whose final task answer can be
parsed with the configured boxed_integer_v1 scorer, retaining both correct and incorrect answers.
Both editors see only text after completed thinking. Unparseable trajectories are archived but
excluded from both editing and likelihood verification. Retained trajectories keep complete original
response IDs and the same full-trajectory mean-logprob measurement (no drift correction).
If all three rollouts are unparseable, consume exactly one skipped iteration, retain the parent,
and make no editor/scoring/candidate-validation calls. No retry, resampling or second iteration.

Unchanged: frozen Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 BF16; context40960;
T0.6/top-p0.95/top-k20, task and editor max8192, thinking enabled; canonical DAPO split seed0
train150/validation300/test300, within-test duplicate rows preserved; initial instruction unchanged;
minibatch1 x3 rollouts; fixed search seed0; max_iterations1; validation1 and test1 per question;
max_task_rollouts3315 as an upper bound; per-trajectory direction verification and nondecreasing
validation admission, rank by validation then signed mean gain. No request-level sampling seed.

Pipeline: fresh base validation300 -> training rollout3 -> parseability filter -> empty pool: skip;
otherwise proposal/measurement/refinement/remeasurement -> full validation300 for each distinct
verified edit -> freeze selection -> initial/selected test300 each (reuse initial if unchanged).
All validation/test rows are scored, including parse failures. No other likelihood condition runs.

Operational: await the previous owned execution's endpoint test and automatic server shutdown,
then start fresh identical vLLM replicas on GPU0/GPU1 with new server manifests and run identity.
These changes affect operational identities only. Previous results and source snapshots are preserved.

Outputs: original captures, editor_evidence.jsonl with inclusion/exclusion records and raw/retained
pool hashes, exact optimizer requests/outputs if any, likelihoods/edits/decisions, complete validation,
frozen selection, test results, costs and source/config identity. This is a single stochastic mechanism
check, not a reliable method effectiveness comparison.
