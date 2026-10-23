# Code map

| Responsibility | Extension point |
|---|---|
| CLI, dry-run, provenance, optimizer-specific selection and test | `src/prompt_policy/cli.py` |
| Standalone TOML validation | `src/prompt_policy/config.py` |
| JSONL split identity and task reward | `src/prompt_policy/benchmarks.py` |
| Local Qwen HF/vLLM rendering, generation and model identity | `src/prompt_policy/backends.py::{LocalHFBackend,VLLMBackend}` |
| Exact token logprob calculation and immutable-response cache | `src/prompt_policy/scoring.py` |
| Question-balanced objective and accuracy guard | `src/prompt_policy/objective.py` |
| Append-only records and cost accounting | `src/prompt_policy/artifacts.py` |

GEPA optimization now uses PromptConsolidation's existing `src/apo/gepa/gepa_artifact/`
and its native `python -m scripts.run_experiments` runner in the separate GEPA environment.
The standalone framework optimizer and `[gepa_refinement]` configuration were removed.
The pinned `vendor/gepa/` snapshot is retained for historical provenance, not imported at runtime.
TextGrad, likelihood editing and archived analyses below remain in this project's environment.

The serial numerical kernel was adapted from the reference listed in `SOURCE_PROVENANCE.json`.
Unlike that Qwen-specific unmasked right-padding path, it does not batch/pad trajectories. Numerical
tests compare it with full causal-LM logits on a tiny random Qwen3 model and chunk-size changes.

Archived TextGrad likelihood audits use `prompt_policy.analysis.textgrad_likelihood`.
The standalone TOML declares source run IDs, the original validation selection and repeat8
evaluation identities, exact model revision, reconstruction rules and GPU. The reader validates
original rewards and parent/proposal/acceptance links, then reuses the same causal hidden-state
alignment as the scorer. It performs no generation and never imports another project.

Completed audits are independently verified by `prompt_policy.analysis.textgrad_report`, which
checks token caches, original/repeated validation attribution and input hashes before writing a
cross-run report. See [TextGrad likelihood analysis](textgrad-likelihood.md) for the statistic,
policy-gradient interpretation, sampling limitations and commands.

Base-to-final selected endpoint comparisons use `prompt_policy.analysis.textgrad_endpoints`.
They fix original base-validation responses, verify and copy exact cached contexts, and score
missing endpoint contexts with the existing numerical kernel. Endpoint identity is checked
against each native summary and the actual accepted parent chain.

Historical DSPy DAPO endpoints use the same endpoint module with explicit
`task_format=dspy_chat_adapter_v1`. `build_legacy_inputs` reads actual captured endpoint messages,
checks identical wrapper/question history, and verifies all saved rewards via `legacy_dapo_score`.
A cache is optional; the no-cache path hashes the pinned model files and scores all contexts.

Historical candidate-origin comparisons extend the same endpoint reader through explicit
`response_origins` and `include_origin_context`. `make_origin_report` separates final-minus-base
from final-minus-origin, preserves accepted/rejected branch identity, and reports each origin
without pooling. Exact cached endpoint score files can be reused after ID/hash checks.

Online TextGrad prompt search extends the existing CLI using a `[textgrad]` configuration section.
`textgrad_optimizer.py` supplies only local forward capture, numerical string losses, an EngineLM
adapter, and admission/selection; upstream `LLMCall.backward`, `StringBasedFunction`, `sum` and
`TextualGradientDescent` are unchanged and verified against `vendor/textgrad/UPSTREAM.json`.
`objective.py::fixed_success_gain` retains failed rollouts in the per-question denominator and
`textgrad_eligibility` enforces positive sequence gain plus a validation-accuracy guard. The benchmark
module owns optional historical DSPy rendering/scoring; reflection requests always use plain messages.


The likelihood-editing loop uses the existing CLI with `[edit_refinement]` and
`[objective] mode="directional_likelihood"`. `edit_optimizer.py::search` measures base validation,
then performs train rollouts, one instruction proposal, fixed-ID measurement, one refinement and
remeasurement. Each distinct likelihood-verified edit receives complete validation; require nondecrease
against the stored parent accuracy, rank by validation then train signed gain, or keep parent.
Optional `search.base_validation_run` reuses a complete prior baseline through
`edit_optimizer.py::reuse_base_validation`. It verifies source configuration digest, benchmark identity,
initial prompt, generation/context/repeat settings, actual model file hashes/dtype/thinking/backend/version,
all expected validation pairs, exact freshly rendered prompt IDs, recomputed rewards and stored summary.
Only then copy the baseline records with explicit source attribution and hashed provenance; charge zero
new generation calls. Source lookup is explicit in standalone TOML, never inferred from similar names.
Missing or mismatched evidence raises rather than silently generating a replacement baseline.
The last accepted prompt is frozen before CLI test. Validation accuracy remains task accuracy.
`evaluate_validation` streams full raw responses/IDs to validation.jsonl, records complete summaries
and progress, and never exposes validation examples to the editors. Search defaults to5 iterations,
reserves complete train+up-to-two-validation cost, and records completed iterations/termination reason.

`objective.py::fixed_directional_gain` owns full/response/branch fixed-parent masks, individual direction
checks, question-balanced sequence/mean gain and optional same-question mean centering. Empty masks
and homogeneous centered reward groups cannot verify changes. `scoring.py::Likelihood` optionally
retains vocabulary top1 LPs; exact full response IDs and per-token LPs remain unchanged.
`LocalHFBackend.response_token_indices` reads thinking token boundaries without retokenization.
`benchmarks.py::parse_answer` is the same extraction used by task reward; it never filters by
correctness. `Benchmark.editor_response` checks that extraction and returns only text after the last
closed `</think>` (or the visible response when no thinking was opened). `edit_optimizer.py` archives
all captures, then retains only parseable trajectories for both editors and fixed-ID verification.
Parseable wrong answers remain failure evidence. `editor_evidence.jsonl` records included/excluded
identities/counts and raw/retained pool hashes. Empty pools consume one iteration and the captured
generation cost but invoke neither editor, likelihood scoring nor candidate validation; no retries or
resampling. Retained `Trajectory` objects keep their original complete response IDs and text; the
existing full/response/branch measurement setting controls masks independently of the editor view.
`edit_prompts.py` renders labeled plain-text evidence; refiner receives mean LP units, per-response changes and the raw/centered criterion.
Scoring scope remains in configuration/artifacts, not the editor message. Both editors emit one instruction tag block; only its body is parsed.
Invalid/incomplete editor output fails without fallback/retry. Numerical feedback names each response and its measured increase, decrease or unchanged value;
semantic diagnosis is generated by the refiner. See [the current pipeline](contrastive-edit-refinement.md)
and [comparison plan](../experiments/directional_validation_full_raw/plan_report.md).

The removed candidate-first contrastive/directional design is not maintained as an extra code path.
Completed historical runs keep their exact source snapshots and artifacts; old complex recipes require
those snapshots and are rejected by the current minimal schema. Existing TextGrad paths and
vendored snapshots are independent and unchanged. `top_k` is configurable for task/reflection sampling;
T=1 full-vocabulary likelihood measurement remains distinct from the sampling policy.
See [the minimal pipeline](contrastive-edit-refinement.md) and
[the current condition](../experiments/directional_edit_dapo_pilot/plan_report.md).

Current integer tasks use paired `benchmark.task_format` / `scorer` values
`boxed_integer_v1`. `task_messages` places only editable P in SYSTEM and the immutable
`BOXED_INTEGER_RULE` followed by `\n\nQuestion:\n` and the original question in USER.
Both HF/vLLM generation and teacher forcing use this renderer. `extract_boxed_integer` reads only
the last balanced `\boxed{...}` outside completed thinking and accepts a signed ASCII integer body.
Earlier boxes may contain intermediate results; missing, unfinished or noninteger final boxes yield
reward0, with no Answer/tag/plain-number fallback. Original full response IDs, including thinking,
formatting and sampled EOS, remain likelihood targets. The boxed protocol and fixed rule enter
benchmark fingerprints and CLI preview/manifest. Historical answer_integer_v1, answer_tag_integer_v1/v2
and dapo_dspy_v1 retain their original rendering/scoring/fingerprints; archived runs are unchanged.
Both editors request general lessons applicable to similar problems as reusable reasoning or
verification strategies, placed after the observed responses.
Only the `<instruction>...</instruction>` body is parsed; this boundary check does not guarantee
semantic generality. Validation remains the empirical check of an otherwise verified candidate.



Local vLLM serving (including optional same-model local replicas) reuses `backends.py::VLLMBackend` through optional `[serving]` in the
existing standalone TOML/CLI. `generate_batch` submits independent rollout requests concurrently
and preserves example/repeat order; the HF fallback remains serial. The same helper batches final
test requests after prompt selection is frozen. No new CLI or optimization component is introduced.
The backend renders the pinned tokenizer's chat template locally, supplies exact integer prompt IDs
to `/v1/completions`, and requires actual returned completion IDs including EOS. Reflection uses plain rendering; an empty system argument omits that message entirely. The local serving manifest and live model path/context/version are
checked and recorded. Generation uses configured temperature/top-p/top-k. Teacher forcing sends
prefix + fixed response IDs with `prompt_logprobs=0`, T=1, top-p=1 and top-k=-1; it selects the
actual response token at positions starting at prefix length. One mandatory dummy continuation is
discarded. No text retokenization, response truncation, sampling seed or API retry is introduced.
`Recorder` protects concurrent generation cost increments. `progress.json` and stdout expose the
current scientific phase; original detailed JSONL records remain the scientific evidence.

The canonical full DAPO test includes one whitespace-normalized duplicate question pair with distinct
IDs. `benchmark.allow_within_split_question_duplicates=true` preserves canonical rows explicitly,
records duplicate identities, and still rejects cross-split question overlap and all duplicate IDs.
The default remains strict. Canonical data exports and archived-ID mappings live with input artifacts,
not as runtime imports of another project.


`edit_refinement.mode="per_example_likelihood_probe"` selects `edit_optimizer.py::probe_search`
through the existing CLI. It keeps `prompt_policy="fixed"`, captures each minibatch once,
then proposes/measures/refines/measures independently for each question. `capture_training_batch`
is shared with validation-gated search. No validation or test is generated and no candidate adopted.
Empty evidence groups and `InvalidInstruction` outputs are recorded per group without retries;
transport/scoring/integrity errors still fail the run. `probe_groups.jsonl` and `probe_summary.json`
separate evidence coverage, valid editor outputs, measured/verified edits, distinct edits, group yield
and refinement rescues. Alignment denominators are measured outputs, not all sampled questions.


Likelihood editor calls use one USER message. `edit_prompts.py` starts with the complete
`PROPOSER_TEMPLATE` and `REFINER_TEMPLATE`, with named placeholders showing the final message order.
`request_payload` dispatches to `render_proposer_request` or `render_refiner_request`. Both fill
`render_evidence` (input/gold once per example, then labeled responses); refiner evidence adds
`render_measurement` immediately below each response, followed by `render_feedback`.
The introductory and final-instruction constants used by CLI previews and provenance hashes are
derived from these templates, not assembled from separate wording fragments. Drift corrections, when enabled, are grouped
with the corresponding response; feedback distinguishes drift-relative changes from absolute changes.
`edit_prompts.py::response_feedback` states which response was reinforced or suppressed, explicitly
flags increases of failures/decreases of successes, and reports unchanged or unmeasurable changes
without inventing an increase/decrease. No generic direction-verification verdict is shown.
No numeric reward lines or separate fixed task-output rule are included in the editor request.
`edit_optimizer.py::optimizer_call` sends that assembled string once, without prepending another
instruction block. `benchmarks.py::task_messages("", request)` omits SYSTEM for plain reflection;
both HF and vLLM use it. `optimizer_calls.jsonl` records exact `messages`, the complete `request`
and `system_prompt=null`; CLI previews expose introduction/instruction text and assembly order.
Task rendering/scoring and other optimizers' reflection system instructions are unchanged.


`edit_refinement.mode="validation_gated_per_example"` extends the existing validation-gated search:
all questions in each minibatch roll out under one parent, then each independently proposes/refines
using only its own retained responses. Invalid editor output skips that group/phase, with no retry;
punctuation-only instruction bodies are invalid. `proposal_groups.jsonl` records coverage and errors.
Every distinct likelihood-eligible instruction is validated once per round (first qualifying source
in schedule order). Select at most one validation-nondecreasing winner by accuracy, then its source
question's signed likelihood gain; the next round rolls out under that winner. Budget reserves up
to two validations per question. The original joint-minibatch mode remains available unchanged.
`reuse_base_validation` accepts a complete phase from a complete or interrupted source run, after
all existing identity/record/summary checks. Provenance records source status and phase-only reuse;
running/failed sources and incomplete baselines are rejected. No source manifest is relabeled.



Archived sampling-policy audits use `python -m prompt_policy.analysis.sampling_policy --config
<analysis.toml> --output <run-dir>`. This CPU-only stage reweights saved top-k raw logprobs
with the source rollout temperature, top-k and top-p. It preserves original response IDs,
checks parent prefix/model/policy identities and writes a new run without touching source
artifacts or calling a model. Missing cutoff ties are bounded using residual vocabulary mass;
nucleus-boundary support is marked ambiguous where needed. Finite archive-only scores are
explicitly nominal. Zero-probability events stay zero; undefined -inf differences are not
reported as gains. It checks parent replay consistency before any update interpretation.

Optional `[serving] rollout_logprobs="processed_logprobs"` captures generation-time
processed sampled-token logprobs. Launch the corresponding vLLM server with
`--logprobs-mode processed_logprobs` and preserve that command in its server manifest.
`validate_rollout_logprobs_mode` rejects mode/startup-command conflicts; an omitted
server flag means raw_logprobs. `raw_logprobs` capture is also available and explicitly labeled.
The default remains no capture for existing configurations. Capture requests use `logprobs=0`
and `return_tokens_as_token_ids=true`, require ID/order/length agreement including sampled EOS,
and reject missing/nonfinite/positive/API-clipped values. `Generation` and `Trajectory` retain
`rollout_token_logprobs`, `rollout_logprobs_mode`, and `rollout_sampling`; training capture in
both edit/TextGrad paths copies them, and validation/test records serialize Generation fields.
Old trajectories load with null metadata. Reflection calls do not request these rollout statistics.
Teacher-forcing scores and cache meaning remain full-vocabulary T1. For pinned vLLM0.11.1,
prompt logprobs are computed separately from unprocessed logits even with processed generation
logprobs enabled. Engine-wide processed mode can change the sampler execution path and cost;
it does not change the requested policy settings. No running server is restarted by this option.

The same sampling-policy audit optionally accepts `[branch_analysis] probability_threshold=0.95`.
`distribution_statistics` compares T1-full, T1-top-k, rollout-temperature/top-k, and the original
rollout policy including top-p on identical saved prefixes. Bounds account for missing cutoff ties;
position-level probabilities, masks, support sizes and before/after counts are saved separately
from fixed-trajectory likelihood scores. This does not change online branch masks or adoption.
Each audit now archives its source code snapshot alongside its standalone config.
