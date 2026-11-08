"""Experimental ALFWorld UCF v2.1 scoring contracts (no training acceptance).

Exact action events include the closing fence. Without a terminator, commands
like 'take apple 1' and 'take apple 10' need not be mutually exclusive events.
"""
from __future__ import annotations
import math
import random

def first_fence_prefix(history_ids, response_ids, fence_id):
    """Stop at the first sampled opening fence; never select the last valid one.

    A response which never reaches a fence has no action-decision node. No
    suffix validity, final action, reward or closing-fence test is used here.
    """
    try:
        end=list(response_ids).index(fence_id)+1
    except ValueError:
        return None
    return list(history_ids)+list(response_ids[:end]), end

def validate_action_tokens(token_sequences):
    seqs = [tuple(s) for s in token_sequences]
    if any(not s for s in seqs) or len(set(seqs)) != len(seqs):
        raise ValueError("Empty or duplicate action token sequence")
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i != j and len(a) <= len(b) and b[:len(a)] == a:
                raise ValueError("Action events overlap: include closing fence and recheck prefix-freeness")

def action_token_sequences(tokenizer, actions):
    # Prefix is ACTUAL sampled ids ending immediately after the first opening
    # fence; it is never decoded and re-tokenized. Canonical action payloads are
    # newly proposed strings, so only those suffixes are tokenized here.
    seqs = [tokenizer.encode(a + "```", add_special_tokens=False) for a in actions]
    validate_action_tokens(seqs)
    return seqs

def score_actions(engine, prefix_ids, token_sequences):
    """Score unrestricted complete action events via vLLM prompt_logprobs.

    One throwaway generated token is required by vLLM's generate API and must
    be charged to compute, but is not appended to training data.
    """
    from vllm import SamplingParams
    if not prefix_ids or any(not isinstance(x,int) for x in prefix_ids):
        raise ValueError("Scoring prefix must be actual nonempty integer token ids")
    validate_action_tokens(token_sequences)
    jobs = [{"prompt_token_ids": list(prefix_ids) + list(ids)} for ids in token_sequences]
    outputs = engine.generate(jobs, SamplingParams(max_tokens=1, temperature=0,
                                                  prompt_logprobs=1), use_tqdm=False)
    if len(outputs) != len(jobs):
        raise RuntimeError("Scoring backend returned an incomplete action batch")
    scores = []
    for ids, out in zip(token_sequences, outputs):
        row = []
        for k, token in enumerate(ids, len(prefix_ids)):
            entry = out.prompt_logprobs[k]
            if entry is None or token not in entry:
                raise RuntimeError("Scoring backend omitted the actual prompt token")
            lp = float(entry[token].logprob)
            if not math.isfinite(lp):
                raise RuntimeError("Non-finite unrestricted action logprob")
            row.append(lp)
        scores.append(row)
    return scores

def action_distribution(actions, token_logprobs):
    if len(actions) != len(token_logprobs) or len(set(actions)) != len(actions):
        raise ValueError("Action mapping mismatch")
    if any(not row or any(not math.isfinite(x) or x > 1e-6 for x in row) for row in token_logprobs):
        raise ValueError("Action scores must be nonempty finite unrestricted log probabilities")
    probabilities = [math.exp(sum(row)) for row in token_logprobs]
    total = math.fsum(probabilities)
    if total > 1 + 1e-6:
        raise ValueError("Non-disjoint events or inconsistent full-vocabulary scores")
    return dict(zip(actions, probabilities)), max(0.0, 1.0 - total)

def force_action(actions, token_sequences, token_logprobs, probabilities, allowed, seed):
    indices = [i for i, a in enumerate(actions) if a in allowed and probabilities[a] > 0]
    if not indices:
        raise ValueError("No positive-probability counterfactual action")
    rng = random.Random(seed)
    index = rng.choices(indices, weights=[probabilities[actions[i]] for i in indices], k=1)[0]
    # PPO old_logprob is the UNRESTRICTED per-token probability, not the
    # renormalized Python selection probability.
    return actions[index], list(token_sequences[index]), list(token_logprobs[index])

def require_ucf_acceptance(receipt):
    if not (receipt.get("model") == "Qwen3.5-2B" and receipt.get("version") == "v2.1"
            and receipt.get("status") == "accepted" and receipt.get("source_sha256")):
        raise RuntimeError("ALFWorld UCF remains experimental: 2B v2.1 acceptance is missing")
    # Receipt acceptance is necessary, not sufficient: ALFWorld integration and
    # the OTHER residual-mass estimator also require independent validation.
