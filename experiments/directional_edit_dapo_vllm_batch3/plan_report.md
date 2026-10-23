# Local vLLM likelihood-editing pilot

User-approved conditions (2026-10-19): minibatch 3 x one rollout, 8192-token task/optimizer
caps, 12 outer steps, at most 24 measured edits; first 32 archived train questions. Sampling
T=0.6, top-p=0.95, top-k=20, thinking enabled. Search RNG 0; no request-level sampling seed.

Base execution identity: `20261019T012552Z-contrastive-edit-dapo-pilot-gpu1`, which used an older
mixed-pair method and made no updates. This run tests the current minimal method, so it does not
isolate the effect of changing the serving backend or minibatch size.
Data source: `20260917T062343Z-dapo-textgrad-ctx40-bbd2/native/splits.json`.
The first 32 train examples are selected in archived order without outcome filtering. Validation
and test preserve the existing first six examples of each split. Exact hashes/IDs live in
`artifacts/inputs/dapo_seed0_train32_test6/manifest.json`. Validation is loaded for split identity
and disjointness but never evaluated or used for update selection.

Frozen Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, native
40960 context; one vLLM 0.11.1 server on physical local GPU1. GPU0 was used temporarily for a short
HF numerical cross-check and released before research execution. Same server handles task,
proposer/refiner and full-vocabulary T=1 fixed-ID teacher forcing. API concurrency 3. Eager mode,
no prefix caching, model generation defaults disabled, no sampling seed. Server launch/identity/log:
`artifacts/servers/20261019-directional-edit-vllm-gpu1/`.

Each step: rollout -> reward -> plain-text proposer -> exact-ID measurement -> plain-text refiner
with parent means and candidate mean deltas -> remeasurement -> adopt the better eligible edit.
An edit is eligible only if every observed success increases and every observed failure decreases.
Its ranking score is the equal-question mean signed sequence-logprob change. No epsilon, validation
guard or test feedback. The fixed final-line Answer integer rule is outside editable instruction.
The final selected prompt is the last accepted prompt, frozen before initial/selected test.

The existing pass-without-replacement scheduler produces batch sizes
3,3,3,3,3,3,3,3,3,3,2,3: 35 train rollouts, not 36 (the 11th batch exhausts the 32-question pass).
Budget 36 is a cap. With test one rollout per question: six initial test outputs, plus six selected
if the prompt changed; exact initial reuse if unchanged. At most 24 optimizer generations and
105 uncached trajectory/context scores (three contexts per trajectory). Parent cache reused
between candidate/refinement; one unscored dummy continuation per scoring request.

Report: per-step rewards/token lengths, per-edit correct-direction counts, per-trajectory mean
changes, signed score, admission/parent chain; accepted-edit examples and frozen endpoint tests.
No performance-improvement claim is supported by six test questions. Numerical cross-check of
11 generated tokens against HF: maximum |difference| 0.0026340, mean 0.00035069 nats/token;
repeated identical vLLM scoring difference 0. IDs/EOS/alignment retained. BF16 engine differences
mean this is not bit-identical to earlier HF likelihood measurements.
