# Auditing archived TextGrad updates

The question is whether actual TextGrad prompt edits increased the likelihood of responses that
received positive reward under their actual parent prompt. This is a diagnostic of the recorded
optimization path, before testing a new optimizer that enforces likelihood ascent.

For a parent prompt p, proposal p', and fixed parent-generated responses y_i, the primary statistic is

```
G(p', p) = mean_i r_i [log pi_p'(y_i | x_i) - log pi_p(y_i | x_i)].
```

`r_i` is the archived binary task reward. A failed response contributes zero. The denominator
includes all questions, rather than only successes. Preserved-response logprob sums all reconstructed token logprobs;
token-mean logprob is a separate length-normalized diagnostic. Every response is scored under
both prompts with identical reconstructed response IDs and its original question. There are no
new rollouts, updated model weights, or test-set inputs.

Compute G separately for the actual train batch that produced the textual feedback and for the
entire original parent validation capture. Include rejected proposals and preserve their actual
parents; a rejected proposal is not the parent of the next edit. Compare G to original validation
accuracy changes and the original accept/revert decisions. Original validation participated in
optimization, so these comparisons do not establish held-out generalization. The separate
historical repeat8 validation evaluation is supplemental evidence and did not select the path.

## Relationship to policy gradients

For a differentiable policy and fixed on-policy samples, the vanilla score-function estimator
is the gradient of the empirical objective `mean_i A_i log pi_theta(y_i | x_i)`, where advantages
are held constant. Its finite difference between two prompts, with A=r and no baseline, is G.
This gives a useful prompt-search objective, but discrete prompt edits do not themselves become
parameter gradients. PPO instead uses probability ratios and clipping; G is not its clipped loss.
See [Schulman et al., equations 1–2 and 7](https://arxiv.org/pdf/1707.06347).

The probability must match the policy being defined. If pi is the raw model distribution, score
full-vocabulary `log_softmax(logits)` at T=1. If pi is a tempered full-vocabulary distribution,
score `log_softmax(logits / T)`. If pi includes top-k/top-p sampling, define its retained support
S_p(h) at each prefix h and score the renormalized distribution

```
q_p(v | h) = 1[v in S_p(h)] * exp(z_p(v,h) / T)
             / sum_{u in S_p(h)} exp(z_p(u,h) / T).
```

Support and normalization change when the prompt changes. An observed token excluded by the
proposal has q=0 and log q=-infinity. The top-k/top-p order and boundary-tie convention depend on
the implementation. Scoring raw-model probabilities for samples from a truncated distribution
does not automatically give an unbiased on-policy gradient estimator for the raw model policy.
Defining raw-model training probabilities is possible, but the sampling mismatch needs explicit
treatment. Hard truncation can also remove support needed by importance corrections.

This audit measures T=1 and T=0.6 full-vocabulary probabilities, and a sequential top-k=20,
top-p=0.95 support diagnostic at T=0.6. It deliberately leaves filtered logprob undefined when
the cutoff is tied. Filtered values are not certified historical sampling-policy probabilities.
The old artifacts did not preserve original output token IDs, stripped stop tokens, or enough
backend identity to claim exact sampler replay. No missing EOS is appended. Reported likelihoods
therefore describe reconstructed preserved response prefixes. Token count agreement alone
cannot establish token identity. Normal-stop/count-matched sensitivity is reported separately.

Even with exact on-policy sampling, a negative finite-step G need not imply lower accuracy: a
prompt may replace one successful reasoning style with another. This audit tests whether ordinary
TextGrad updates satisfy the proposed likelihood-ascent condition, not whether enforcing that
condition will improve accuracy. A new optimization comparison is needed for the latter claim.

There is a precise population interpretation for binary reward. If sampling and scoring use the
same full-support policy, write z=(x,y), P_p(z)=d(x)*pi_p(y|x), J_p for expected reward,
and s_p(z)=P_p(z|r=1) for the joint input/response distribution conditioned on success. For J_p>0,

```
E_P_p[r * log(P_p' / P_p)]
    = J_p * (log(J_p' / J_p) - KL(s_p || s_p')).
```

This follows by substituting `P_p(z) = J_p * s_p(z)` on successful trajectories. Positive
population G is sufficient for reward improvement, while negative G permits improved reward if
the conditional successful distribution changes enough. It penalizes replacing successful modes.
These assumptions do not hold exactly for this finite, filtered-sampling archive scored at T=1;
the identity explains the proposed objective rather than proving anything from these numbers.

## Execution and outputs

Scientific inputs and execution settings are declared in the standalone
`experiments/textgrad_update_likelihood_gpu1/analysis.toml`. The associated plan documents user
authorization and the exact two archived source run IDs. The original projects are read only.

```bash
uv sync --frozen --extra dev --extra local --extra analysis
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m prompt_policy.analysis.textgrad_likelihood \
  --config experiments/textgrad_update_likelihood_gpu1/analysis.toml --dry-run
```

After review and authorization, omit `--dry-run`. The configured output must be new. To recover an
interrupted execution, add `--resume`: the plan, source hash and input identities must match.
Only physical GPU 1 may be exposed by this specific approved configuration.

The run stores the input plan and hashes, reconstruction audit, per-context token NPZ traces and
hashes, append-only score index, per-trajectory changes, per-update metrics, and PNG/PDF figures.
Question bootstrap intervals condition on saved responses; they do not account for regeneration
or independent-run uncertainty. Proposals share parents and validation questions.

After the run completes, independently check every NPZ hash, fixed response identity and token
sum, and audit original/repeated validation requests and reward aggregation:

```bash
uv run --no-sync python -m prompt_policy.analysis.textgrad_report \
  --run artifacts/runs/20261015-textgrad-update-likelihood-gpu1 \
  --report artifacts/reports/textgrad_update_likelihood_gpu1
```

The report includes accepted/rejected sign counts, the three accepted updates, descriptive
within-benchmark correlations, validation request audits and verification evidence. It checks
source hashes again and refuses incomplete runs or an existing report output directory.
