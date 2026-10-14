# PromptPolicyOptimization

Independent prompt search that verifies whether a proposed instruction increases the likelihood
of **the same successful trajectories sampled from its parent**. Model weights remain frozen.
This repository has its own Git history, environment, data interface and artifacts; it does not
import `common`, `mopcd`, `gepa_artifact`, or another workspace project.

## Setup and verification

```bash
uv sync --frozen --extra dev --extra local
uv run --no-sync pytest
uv run --no-sync ruff check src tests
uv run --no-sync prompt-policy experiments/qwen3_success_likelihood/experiment.toml --dry-run
```

The dry-run validates settings, split identities and the vendored GEPA source. It neither loads
a model nor generates responses. The example deliberately leaves the model unspecified and uses
tiny toy data. Fill `model.name_or_path` with a local HF checkpoint directory. For a remote model,
also set `model.revision` to an immutable 40-character commit. Remote weights are downloaded only
when running the experiment. Local weights/tokenizer/config files are hashed for provenance.

Set the three benchmark paths to your own explicitly split JSONL data, with one record per line:

```json
{"id": "train-0001", "question": "What is 2 + 3?", "answer": "5"}
```

All values are strings. IDs and normalized question text must be unique across all splits.
Data and relative model paths are resolved from the TOML's directory. The built-in scorer requires
a final `### answer`, `Answer: answer`, or `\boxed{answer}`. `numeric` compares exact rational
numbers; `exact_match` compares strings after trimming outside whitespace. These are explicit
project scorers, **not claims of equivalence to canonical DAPO/GPQA evaluation**. Implement a new
benchmark's canonical scorer in `benchmarks.py` and test it before interpreting research results.

Run after reviewing the dry-run and replacing the toy data:

```bash
uv run --no-sync prompt-policy experiments/qwen3_success_likelihood/experiment.toml
```

Each invocation creates a new `artifacts/runs/<UTC timestamp>-<unique id>/`. `--output` accepts an
explicit new directory and refuses existing directories. The first version does not resume runs.
No real experiment is started by setup, tests or dry-run.

## Method

1. GEPA selects a parent from its accuracy Pareto frontier and a train minibatch.
2. Generate `train_rollouts` independent responses per question and verify task reward.
3. GEPA reflects on both successes and failures to propose a new instruction.
4. Freeze the parent's successful response token IDs. Teacher-force those same IDs under parent
   and child requests, at scoring temperature **1**, over the full vocabulary.
5. Generate new child rollouts on the same questions to estimate task accuracy.
6. Accept when `delta_accuracy + weight * success_gain > tolerance`, provided the observed
   accuracy drop is within `max_accuracy_drop`. Identical instructions are rejected.
7. Evaluate accepted candidates on validation; GEPA's global scores/Pareto frontier and final
   selection use **accuracy only**. Freeze selection before evaluating the initial and selected
   prompts on test. If selection is the initial prompt, reuse its test evaluation explicitly.

`success_gain` averages token-mean logprob changes inside each successful trajectory, then
trajectories inside each eligible question, then eligible questions. A question with many successes
does not outweigh one with a single success. An empty successful pool contributes zero, allowing
accuracy improvement to drive exploration. Sum logprob change, increase fraction, failure gain and
coverage are also recorded. Failure likelihood is diagnostic only in this version.

The scored response includes **actual sampled EOS tokens**. No EOS or other token is appended.
Display text excludes terminal EOS and preserves thinking delimiters. Parent re-rendering must
match the captured request token IDs. Context overflow is an error rather than silent truncation.
The scorer uses serial causal hidden states and chunked FP32 full-vocabulary log-softmax.

Generation temperature/top-p are independent of T=1 scoring: this is a length-normalized,
success-only search surrogate, not an unbiased policy-gradient estimator or a PPO update.
`search_seed` governs GEPA selection/shuffling. No request-level sampling seed or global training
seed is applied during real inference. Test data never enters reflection or candidate acceptance.

Reflection uses the same frozen local model, serially, to avoid an API credential dependency or a
second resident model. Qwen thinking is enabled by the example; raise reflection's token limit if
it truncates. Truncated reflection causes a visible failed run rather than accepting partial text.

## Comparisons and budget

Make standalone TOML copies; no experiment inheritance or hidden scientific defaults. Useful controls:

| Condition | `mode` | `weight` | `likelihood_feedback` | `measure_likelihood` |
|---|---|---:|---|---|
| Accuracy, success examples | `accuracy` | 0 | false | true |
| Accuracy, likelihood feedback | `accuracy` | 0 | true | true |
| Success-likelihood objective | `success_likelihood` | explicit positive value | true | true |

Keeping measurements enabled in controls makes teacher-forcing overhead visible. This project's
accuracy control uses the same success-aware adapter and disables perfect-score skipping, so it
is not a byte-for-byte reproduction of the paper's original GEPA baseline.

`weight=1` is an illustrative setting, not a calibrated recommendation. Choose it on validation
without looking at test, and report sensitivity. The minibatch accuracy guard is an observation,
not a guarantee of nondecreasing true accuracy. Correct final answers do not verify every reasoning
step, and success likelihood can reward style affinity; measure held-out accuracy and failure gain.

`max_task_rollouts` counts **actual individual task generations**, including validation repeats,
and excludes reflection and the final test. Limits are checked between complete GEPA iterations;
the last iteration can exceed the rollout threshold. Report observed counts, generated/prefill
tokens, teacher-forced tokens, reflection calls and elapsed time. The pinned GEPA's internal
metric-call counter counts validation question slots; it is retained separately and is not the
project's budget authority. There is no new scheduler, service or database.

## Run records

- `config.json`, `manifest.json`: resolved conditions, source/model/data hashes, environment,
  pinned GEPA revision, completion/failure status, time and cost.
- `trajectories.jsonl`: actual task responses, rewards, split, prompt, original request/response IDs.
- `likelihood.jsonl`: token logprobs, sum/mean, cache keys and cache hits for fixed responses.
- `decisions.jsonl`: parent and child instructions, frozen success pool, gains and accept/reject reason.
- `reflection.jsonl`: actual reflection requests and responses including token IDs.
- `gepa/`: GEPA's own native state and lineage, unchanged format.
- `candidates.json`, `selection.json`: candidate pool and validation-only selection before test.
- `test.jsonl`, `result.json`: initial/selected held-out responses, repeated accuracy and costs.

Research outputs are ignored by Git. Source is kept separately from input/output artifacts.
The repository does not upload, delete or mirror scientific artifacts automatically.

See [code map](docs/codemap.md), [experimental design](docs/design.md) and
[third-party provenance](THIRD_PARTY_NOTICES.md).

