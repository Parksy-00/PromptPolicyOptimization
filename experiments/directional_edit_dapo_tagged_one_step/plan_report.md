# One-step tagged-output smoke run

Base run identity: `20261019T063416Z-directional-edit-dapo-vllm-b3-gpu1`. Its recorded config was compared with the source recipe before copying.
User request: run exactly one optimizer step after applying instruction/answer tag contracts.

Requested changes: task renderer/scorer `answer_tag_integer_v1`; both editors require and parse one
`<instruction>` block; max_iterations=1. Derived task budget=3 (one complete 3-question minibatch).
Current tagged editor source was implemented and verified in the previous turn; no further prompt edits.
Operational serving manifest uses a fresh local GPU1 server identity; model/runtime/settings unchanged.

Same Qwen3-8B BF16 revision b968826d9c46dd6066d109eabc6255188de91218, 40960 context,
T=.6 / top-p=.95 / top-k20 / max8192 / thinking enabled for tasks and editors, concurrency3,
search RNG0, original initial prompt, DAPO archived train32/validation6/test6 inputs from
20260917T062343Z-dapo-textgrad-ctx40-bbd2. No request-level sampling seeds.
First minibatch must match the base run's first questions: 8428,4998,1187.

Execution scope: existing `edit_optimizer.search` Python API for exactly one minibatch, with the
existing config loader, VLLMBackend, Recorder and scorer. The standard CLI automatically evaluates
test endpoints; this smoke uses the existing search API so it does not add unrequested test calls.
No new module/CLI command/optimizer path is added. Validation/test files are validated as split
identities but no validation/test generation or accuracy measurement is performed.

Outputs: original task messages/token IDs, parsed answer/reward, both editor system/user/raw output
and extracted instruction, fixed-trajectory likelihood deltas, verified/admission decision,
source/model/input identities, and readable prompt traces. On malformed editor tags the current
fail-fast behavior records the raw output and stops; no hidden retries or extra optimization steps.
