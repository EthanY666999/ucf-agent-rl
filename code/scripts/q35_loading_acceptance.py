"""Fixed, non-training HF/vLLM loading acceptance for the actual F03 actor."""
import hashlib
import json
import math
import random
from pathlib import Path

F03 = Path('/data/runs/stage2/F03_SFT_seed42/sft/final_model')
PROMPTS = Path(__file__).resolve().parents[1] / 'artifacts/f03_loading_prompts.json'
# Predeclared engineering tolerance in natural-log units, never tuned to test scores.
MAX_ABS_ERROR = 0.125
MEAN_ABS_ERROR = 0.02
MAX_NEW_TOKENS = 8


def compare_logprobs(hf, vllm):
    assert len(hf) == len(vllm) and len(hf) > 0
    assert all(math.isfinite(x) for x in hf + vllm)
    errors = [abs(a-b) for a, b in zip(hf, vllm)]
    return dict(passed=max(errors) <= MAX_ABS_ERROR and sum(errors)/len(errors) <= MEAN_ABS_ERROR,
                max_abs_error=max(errors), mean_abs_error=sum(errors)/len(errors),
                max_abs_tolerance=MAX_ABS_ERROR, mean_abs_tolerance=MEAN_ABS_ERROR,
                hf_logprobs=hf, vllm_logprobs=vllm, absolute_errors=errors)


def audit_loaded_f03(actor, ref, engine, tokenizer, cfg, out, selected_logprobs):
    if Path(cfg['model']).resolve() != F03.resolve():
        return None
    import numpy as np
    import torch
    from vllm import SamplingParams
    from patr_core import atomic_json
    raw = PROMPTS.read_bytes()
    fixture = json.loads(raw)
    assert hashlib.sha256((F03/'config.json').read_bytes()).hexdigest() == fixture['model_config_sha256']
    assert len(fixture['prompts']) == 10 and len({p['uid'] for p in fixture['prompts']}) == 10
    assert fixture['max_new_tokens'] == MAX_NEW_TOKENS
    assert fixture['max_abs_error'] == MAX_ABS_ERROR and fixture['mean_abs_error'] == MEAN_ABS_ERROR
    # Preserve training RNG and model mode; probe requests use their own explicit seeds.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    actor_mode, ref_mode = actor.training, ref.training
    receipt = dict(passed=False, model=str(F03.resolve()), prompt_fixture_sha256=hashlib.sha256(raw).hexdigest(),
                   source='same build_models/build_engine/sync_weights path as evaluation and RL',
                   precision='HF frozen actual-F03 bf16 reference; vLLM actual actor synchronized to bf16',
                   prompt_count=10, max_new_tokens=MAX_NEW_TOKENS, comparisons=[])
    try:
        actor.eval(); ref.eval()
        # Independently loaded HF reference must match the actor used to synchronize vLLM.
        for left, right in [(actor.model.embed_tokens.weight, ref.model.embed_tokens.weight),
                            (actor.model.layers[-1].mlp.down_proj.weight, ref.model.layers[-1].mlp.down_proj.weight)]:
            assert torch.equal(left.detach().to(torch.bfloat16), right.detach()), 'Actor/reference weight mismatch'
        for i, prompt in enumerate(fixture['prompts']):
            ids = prompt['input_ids']
            assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == prompt['ids_sha256']
            params = SamplingParams(temperature=0., max_tokens=MAX_NEW_TOKENS, logprobs=1, seed=710042+i)
            output = engine.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0].outputs[0]
            generated = list(output.token_ids)
            assert 0 < len(generated) <= MAX_NEW_TOKENS
            vl = [float(lp[token].logprob) for token, lp in zip(generated, output.logprobs)]
            assert len(vl) == len(generated)
            sequence = torch.tensor([ids+generated], device='cuda', dtype=torch.long)
            mask = torch.zeros_like(sequence); mask[:, len(ids):] = 1
            with torch.no_grad():
                hf = selected_logprobs(ref, sequence, torch.ones_like(sequence), mask).float().cpu().tolist()
            comparison = compare_logprobs(hf, vl)
            receipt['comparisons'].append(dict(uid=prompt['uid'], token_ids=generated, **comparison))
        errors = [e for row in receipt['comparisons'] for e in row['absolute_errors']]
        receipt.update(passed=all(x['passed'] for x in receipt['comparisons']),
                       max_abs_error=max(errors), mean_abs_error=sum(errors)/len(errors),
                       total_tokens=len(errors))
        atomic_json(Path(out)/'loading_logprob_check.json', receipt)
        assert receipt['passed'], 'F03 HF/vLLM logprob acceptance failed; do not relax tolerance automatically'
        return receipt
    except Exception as exc:
        receipt['error'] = repr(exc)
        atomic_json(Path(out)/'loading_logprob_check.json', receipt)
        raise
    finally:
        actor.train(actor_mode); ref.train(ref_mode)
        engine.llm_engine.reset_prefix_cache()
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng); np.random.set_state(np_rng)
