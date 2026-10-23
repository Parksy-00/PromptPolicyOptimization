# One-step parseable evidence check at task max16384

Approved request: increase task max length to the previous16384-token setting and remeasure base
validation under that condition. Base condition reference:
20261019T124942Z-directional-boxed-parseable-reuse-base-one-step-gpu01 (8K search completed,
no parseable train evidence and no update; its remaining unchanged-prompt test was interrupted and preserved).

Requested scientific changes: generation.max_new_tokens8192 ->16384 for all task train/validation/test
responses; remove search.base_validation_run so base validation300 is generated fresh. Old8K base
scores and old training responses are not reused. Reflection output max8192 remains unchanged.

Unchanged: frozen Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 BF16; native context40960;
T0.6/top-p0.95/top-k20 and thinking enabled; boxed_integer_v1 task contract/parser; initial instruction;
canonical DAPO seed0 train150/validation300/test300, original within-test duplicate rows preserved;
one train question x3 rollouts, fixed search seed0 schedule, max_iterations1; full-response mean LP,
no drift correction; validation1/test1; max_task_rollouts3315 as an upper bound. No request seeds.

Pipeline: fresh16K base validation300 -> train3 -> retain answer-parseable correct/incorrect responses
-> editors see post-thinking text only -> full original fixed-ID likelihood checks -> refine/remeasure
-> each distinct verified candidate gets fresh16K validation300 -> admit nondecreasing validation,
rank by validation then signed likelihood gain -> freeze selected prompt -> initial/selected test300
(reuse initial only if unchanged). Empty evidence skips the one iteration with no retry/resampling.
All raw IDs/text and filter inclusion/exclusion records are preserved; editor display does not crop
likelihood targets. This one-step check does not establish reliable method effectiveness.

Operational: two fresh owned vLLM replicas on GPU0/GPU1, same model/context/serving arguments as
before and new manifests. Fixed-ID scoring handles one trajectory per request, so task-prefix+16K
fits native context. Actual editor input depends on post-thinking response lengths; the existing
context guard rejects any oversized request without truncation. No context/rope change is introduced.
