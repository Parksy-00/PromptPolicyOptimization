import math
import random

import pytest

from prompt_policy.analysis.sampling_policy import policy_token, scope_summary


def row(probs):
    return {str(i): {"logprob": math.log(p)} for i, p in enumerate(probs)}


def test_temperature_and_nucleus_renormalize_and_preserve_zero():
    actual = policy_token(row([.8, .2]), 1, temperature=.6, top_k=2, top_p=.95, vocab_size=2)
    expected = .2 ** (1 / .6) / (.8 ** (1 / .6) + .2 ** (1 / .6))
    assert actual["support"] == "retained"
    assert actual["nominal_logprob"] == pytest.approx(math.log(expected))
    excluded = policy_token(row([.98, .02]), 1, temperature=1, top_k=2, top_p=.95, vocab_size=2)
    assert excluded["support"] == "excluded" and excluded["exclusion_reason"] == "top_p"
    assert excluded["nominal_logprob"] is None and excluded["nominal_zero"]
    result = scope_summary([dict(excluded, position=0)])
    assert result["sequence_logprob"] == "-inf"


def test_topk_cutoff_and_missing_ties_are_not_silently_ignored():
    excluded = policy_token(row([.6, .3, .1]), 2, temperature=1, top_k=2, top_p=.95, vocab_size=3)
    assert excluded["exclusion_reason"] == "top_k"
    tied = policy_token(row([.4, .2]), 1, temperature=1, top_k=2, top_p=.95, vocab_size=4)
    assert tied["max_missing_cutoff_ties"] == 2
    assert tied["support"] == "retained"
    uncertain = policy_token(row([.75, .08]), 1, temperature=.6, top_k=2, top_p=.95, vocab_size=4)
    assert uncertain["support"] == "ambiguous"


def test_support_bounds_against_complete_distributions_with_cutoff_ties():
    rng = random.Random(0)
    for _ in range(120):
        logits = [rng.choice([0., -.5, -1., -2., -4., -8.]) for _ in range(30)]
        z = math.fsum(math.exp(v) for v in logits)
        probs = [math.exp(v) / z for v in logits]
        order = sorted(range(30), key=lambda i: probs[i], reverse=True)
        cutoff = probs[order[4]]
        weights = {i: probs[i] ** (1 / .6) for i in range(30) if probs[i] >= cutoff}
        total = math.fsum(weights.values())
        retained, cumulative = set(), 0.
        for i in sorted(weights, key=lambda j: (weights[j], j)):
            cumulative += weights[i]
            if cumulative / total > .05:
                retained.add(i)
        for token in [order[0], order[4], order[5], order[-1]]:
            archive = {str(i): {"logprob": math.log(probs[i])} for i in set(order[:5]) | {token}}
            result = policy_token(archive, token, temperature=.6, top_k=5, top_p=.95, vocab_size=30)
            if result["support"] == "retained":
                assert token in retained
            elif result["support"] == "excluded":
                assert token not in retained


def test_branch_mask_shrinks_with_temperature_and_top_p():
    from prompt_policy.analysis.sampling_policy import distribution_statistics

    probs = [.93, .069, .001]
    assert max(probs) < .95
    params = dict(top_k=2, vocab_size=3, probability_threshold=.95)
    raw = distribution_statistics(row(probs), temperature=1, top_p=1, **params)
    sharp = distribution_statistics(row(probs), temperature=.6, top_p=1, **params)
    nucleus = distribution_statistics(row(probs), temperature=.6, top_p=.95, **params)
    assert raw['branch_nominal'] and not sharp['branch_nominal']
    assert sharp['support_size_nominal'] == 2  # Confident is not literally deterministic.
    assert nucleus['support_size_nominal'] == 1 and nucleus['pmax_nominal'] == 1


def test_pmax_bounds_cover_missing_cutoff_ties_and_nucleus():
    from prompt_policy.analysis.sampling_policy import distribution_statistics

    rng = random.Random(1)
    for _ in range(100):
        logits = [rng.choice([0., -.5, -1., -2., -4., -8.]) for _ in range(30)]
        z = math.fsum(math.exp(v) for v in logits)
        probs = [math.exp(v) / z for v in logits]
        order = sorted(range(30), key=lambda i: probs[i], reverse=True)
        archive = {str(i): {'logprob': math.log(probs[i])} for i in order[:5]}
        for top_p in [1., .95]:
            value = distribution_statistics(archive, temperature=.6, top_k=5, top_p=top_p,
                                            vocab_size=30, probability_threshold=.95)
            cutoff = probs[order[4]]
            weights = sorted(p ** (1 / .6) for p in probs if p >= cutoff)
            total = math.fsum(weights)
            kept, cumulative = [], 0.
            for weight in weights:
                cumulative += weight
                if cumulative / total > 1-top_p:
                    kept.append(weight)
            actual = max(kept) / math.fsum(kept)
            assert value['pmax_lower'] - 1e-10 <= actual <= value['pmax_upper'] + 1e-10
