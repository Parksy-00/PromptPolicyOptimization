# Implementation verification

Verified in the independent project environment: Python 3.12.11, PyTorch 2.14.1,
Transformers 5.19.0, vendored GEPA 0.1.4 at the revision in `vendor/gepa/UPSTREAM.json`.

- `uv run --no-sync pytest -q`: **36 passed**.
- `uv run --no-sync ruff check src tests`: passed.
- `uv run --no-sync ruff format --check src tests`: passed.
- CLI dry-run: validates all split/source identities and reports the missing model without
  creating a run or loading a model.

The tests exercise the real GEPA engine's acceptance hook, accuracy-only global scores,
accuracy ties, guard rejection, immutable parent trajectories, per-question weighting,
cache/token alignment, train/test separation and complete/failed run records.

Numerical tests compare a tiny randomly initialized Qwen3 model's teacher-forced token scores
against complete causal-LM logits on CPU and CUDA, across head chunk sizes. A separate tiny
locally saved checkpoint/tokenizer exercises real HF loading, generation, token IDs, EOS,
thinking begun in the request, model identity and context-overflow refusal. No model downloads,
research dataset runs, or pretrained Qwen3-8B experiments were used for these tests.

The supplied experiment is a template using toy arithmetic data with an unspecified target
model. Research efficacy, canonical DAPO/GPQA scoring and objective-weight calibration remain
unmeasured. Configure and review real scientific conditions before launching an experiment.


## Candidate-first contrastive refinement (2026-10-17)

Implementation review validation, with GPU visibility disabled:

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q`: **85 passed, 1 skipped**;
  the skipped case is the existing CUDA numerical test.
- `uv run --no-sync ruff check src tests`: passed.
- `uv run --no-sync ruff format --check src tests`: passed.
- New DAPO standalone TOML dry-run: passed; includes both optimizer prompts, exact split
  fingerprints, question schedule and per-iteration bounds. No model load or run directory.

Added controlled tests exercise failed-relative gains, suppression of successful responses,
question weighting, sum-versus-mean disagreement, full-context evidence, likelihood masking,
parent/reference refresh, no-mixed skips, strict budgets, actual admission/final selection,
malformed/truncated optimizer output, ablation feedback in the next round and exact prompt exports.
New optimizer integration tests use deterministic fake backends; existing tiny-model tests
build their own random CPU models. No pretrained Qwen research generation or GPU run occurred.
The draft pipeline/scientific conditions await the user's requested design review.


## Fixed Answer integer interface (2026-10-19)

User decision: freeze the last-line `Answer: <integer>` rule and optimize instruction P only.
The new `answer_integer_v1` renderer/scorer pair replaces the DSPy format only in the draft
contrastive DAPO condition. Historical configurations, TextGrad source and vendor source are preserved.

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q`: **106 passed, 1 skipped** (CUDA-only case).
- Ruff lint and formatting checks: passed.
- Updated standalone DAPO dry-run: passed without model load or research generation.
- Tests cover signed/leading-zero integers, wrong answers, final-line extraction, thinking in
  completion or request, malformed/boxed/fraction answers, immutable rendering outside P,
  generation/scoring request alignment, plain reflection requests, identity and end-to-end admission.

Only the fixed output interface and editable scope are approved decisions; the broader research
run remains pending review. No pretrained Qwen or GPU research execution was launched.

## Reward-directed refinement revision (2026-10-19)

User annotations requested Qwen thinking sampling T0.6/top-p0.95/top-k20, a final-line plain-integer
Answer rule, an explanation of the unsupported 1e-6 cutoff, and verification with failures only,
successes only, or both. The new directional_likelihood condition uses zero sign cutoff and checks
each captured trajectory; it ranks signed gains with equal question weight. The completed mixed-only
pilot and its source snapshot remain intact. No revised pretrained Qwen/GPU research run was started.

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q`: **129 passed, 1 skipped** (existing CUDA-only case).
- `uv run --no-sync ruff check src tests`: passed.
- `uv run --no-sync ruff format --check src tests`: passed.
- Revised standalone DAPO dry-run: executable, no missing conditions, no model load or research generation.
- Controlled integrations cover all three reward compositions, no fabricated missing classes,
  single-rollout support, strict individual signs even with favorable averages, neutral rejection,
  tiny signed gains with zero cutoff, equal question weights, validation guard, and both CLI modes.
- Real local backend tests use tiny random CPU checkpoints and verify top_k=20 reaches task/reflection
  generation calls, while older TOMLs retain top_k=0. Six directional request fixtures are captured
  from the deterministic toy backend and explicitly labeled synthetic.

## Minimal loop replacement (2026-10-19)

The user requested removal of the overbuilt editing components. The current path now makes one
semantic instruction proposal and one likelihood-informed refinement per training minibatch.
It returns full instruction text, measures the exact original response IDs, keeps the existing
individual success/failure direction check, and adopts the better verified improvement. Validation
filtering/selection, hypotheses, patch schemas, ablations, branches, windows and history are removed.
The former complex editing modes are not retained as fallback implementations. Historical run
artifacts/source snapshots remain unchanged; the separate GEPA/TextGrad paths remain supported.

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q`: **106 passed, 1 skipped** (CUDA-only case).
- Ruff lint/format checks: passed.
- Minimal standalone DAPO dry-run: executable=true, missing=[], two optimizer calls and two
  measurements per step; final selection is last_verified_prompt, with validation accuracy null.
- Controlled tests cover all reward compositions, fixed response IDs, measured refiner inputs,
  better-proposal fallback, unchanged parent on failed directions, next-step parent refresh,
  task budget, incomplete output preservation, individual sign verification, question weighting,
  no validation in search, and CLI selection before held-out test.
- Removed component-specific tests were replaced with the minimal loop tests. No pretrained-model
  generation or GPU research execution was launched for this implementation change.

## Semantic likelihood-shift refiner (2026-10-19)

The refiner now receives labeled plain text in the requested order: current P, examples and full
successful/failed trajectories, parent mean/sequence logprob, candidate P', and per-response changes.
Its instruction asks it to infer which semantic component could have caused an undesirable shift
and revise the instruction to reverse it. No hypothesis output field or separate stage was added.
The proposer prompt was preserved. Mean logprob diagnostics were added using existing scorer outputs;
sequence-based verification and selection are unchanged. Examples are explicitly synthetic.

- Whole CPU suite: 108 passed, 1 skipped (CUDA-only case).
- Ruff lint/format and model-free dry-run: passed.
- Regression tests reproduce the supplied -0.42/-0.31, +0.18/+0.24 example and verify rejection,
  full trajectory content, human-readable refiner input, and missing-class handling.
- No pretrained-model generation or GPU research run was launched.

## Plain-text proposer and mean-only optimizer input (2026-10-19)

Both proposer and refiner now receive labeled plain text from the same renderer. The proposer
receives task/reward evidence without likelihood numbers; the refiner additionally receives only
parent mean logprob and per-trajectory mean-logprob changes. Sequence values remain in raw records
and internal verification/ranking, and are not displayed in optimizer input.
The bottom feedback remains predefined: computed verification selects one of two fixed sentences.
It does not generate semantic explanations; those are inferred by the refiner.

- CPU suite: 108 passed, 1 skipped (CUDA-only case); Ruff lint/format and dry-run passed.
- Integrations verify both text input forms, absence of sequence fields, the supplied mean values,
  preset feedback selection for correct/wrong directions, and single-class evidence.
- Both synthetic examples are exported as .txt; obsolete proposer JSON fixture removed.
- No GPU research generation was performed.


2026-10-19 local vLLM backend: `uv run --no-sync pytest -q`: 118 passed (11.60s);
`uv run --no-sync ruff check src tests`: passed. HTTP contract tests cover exact prompt/response
IDs, retained EOS, causal teacher-forcing offsets, full-vocabulary scoring parameters, missing
fields/context overflow rejection, concurrent rollout order/accounting, and local-only configuration.
Current user-edited refiner system was preserved; exported example and assertion updated to match.
Live API smoke and short HF comparison are recorded under
`artifacts/servers/20261019-directional-edit-vllm-gpu1/api_smoke.json`.


Local authorized run `20261019T063416Z-directional-edit-dapo-vllm-b3-gpu1` completed in
2394.90s: train35/editor24/uncached scoring102; 13/24 edits verified, 8 updates, 3 refiner rescues.
All admissions were failure-only; mixed batches0/3. Strict test3/6 initial,2/6 selected. Editor
outputs copied solutions/answers and introduced formatting conflicts, so this is a pipeline pilot,
not proof of instruction optimization effectiveness. Exact-ID/score/lineage/source audit passed.
Run report, plots and all assembled editor prompts are saved inside its artifact directory.


2026-10-19 user-requested instruction/answer tags: `uv run --no-sync pytest -q`: **168 passed**
(11.60s); Ruff check/format passed. New tests cover tagged extraction with surrounding reports,
missing/empty/duplicate/nested/malformed tags, thinking boundaries/prefix, rejection of unwrapped
and boxed answers, old/new fingerprint separation, HF task/teacher-forcing rendering, and CLI
selection/test using the answer-tag protocol. The unexecuted minimal pilot dry-run is executable
and previews both output contracts. Completed-run inputs/config/artifacts are preserved. No new
GPU scientific run or serving process was launched for this change.


2026-10-19 task rule placement: `answer_tag_integer_v2` puts only P in system and the immutable
answer rule then `Question:`/question in user. Legacy v1 renderers/fingerprints remain available.
`CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q`: **169 passed, 1 skipped** (CUDA-only, 11.81s).
Ruff check/format and updated pilot dry-run passed. Tests check exact role/message ordering, fixed
user content across candidate prompts, unchanged answer parsing, distinct protocol fingerprint,
shared generation/teacher-forcing prefix, and unaffected plain editor rendering. No new GPU run.


## 2026-10-19 validation gate and likelihood scopes

CPU suite:185 passed,1 CUDA-only skipped; Ruff format/check passed. Four full-scale standalone
comparison configurations pass the existing CLI dry-run (validation300,5 iterations, budget6615).
Tests verify validation rejection despite likelihood improvement, validation-first ranking, no validation
for failed likelihood edits, validation/test separation, five-iteration default and budget termination.
Numerical tiny-Qwen tests compare vocabulary top1 LP against full causal logits. HTTP tests verify
actual-token/top1 causal alignment, replica distribution and streaming result identities.
Fixed-mask tests cover response-only context retention, empty masks, parent-fixed branch positions,
question centering and homogeneous-group abstention. Canonical duplicate preservation stays opt-in
and cannot allow cross-split question overlap.

Both actual local vLLM replicas passed a synthetic18-token teacher-forcing transport check:
actual-token LP with prompt_logprobs0 versus1 difference0; GPU0/GPU1 LP difference0 on this fixture.
This is not a research generation or evidence of method effectiveness. The numerical test artifact is
`artifacts/reports/validation_likelihood_design_preflight/two_gpu_api_check.json`.
Research launch remains pending full-split scope confirmation; no300-question validation has run yet.
