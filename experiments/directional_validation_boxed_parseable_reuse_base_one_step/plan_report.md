# Parseable post-thinking evidence, one step, reused base validation

Approved by the user: rerun the parseable/post-thinking implementation for one iteration and avoid
repeating an already measured matching base validation. Base source run:
20261019T103742Z-directional-boxed-full-raw-one-step-gpu01, complete baseline136/300 (45.3333%).
The interrupted fresh-base attempt20261019T123529Z is preserved; its12 partial base responses are
not used. No training rollout from prior executions is reused.

The existing CLI/search path verifies and copies the explicit source baseline with source hashes,
matching initial prompt, generation/context/repeat settings, benchmark identity, actual model file
hashes/dtype/thinking/backend/version, all expected validation responses, exact rendered prompt IDs,
rescored rewards and stored summary. Any mismatch raises. Base generation costs zero new calls.

Other settings remain approved and unchanged: frozen Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218
BF16; context40960; T0.6/top-p0.95/top-k20/max8192 for task/editor; thinking enabled; canonical DAPO
split seed0 train150/validation300/test300; one question x3 rollouts, search seed0, max_iterations1;
full-trajectory mean LP, no drift correction; validation1/test1; max_task_rollouts3315 upper bound.
Only answer-parseable training trajectories enter editing and fixed-ID verification; editors see
post-thinking text, while retained full original IDs remain measurement targets. Valid wrong answers
are kept. Empty evidence consumes one skipped iteration with no retry/resampling/editor/scoring or
candidate-validation calls. All captures/exclusions are archived. Existing validation admission and
frozen initial/selected test remain unchanged. Validation/test never enter editor input.

Operations: fresh owned vLLM replicas on GPU0/GPU1 with identical serving arguments and new manifests;
new run/config/source identity. No other likelihood condition or extra iteration is scheduled.
