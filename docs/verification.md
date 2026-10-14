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

