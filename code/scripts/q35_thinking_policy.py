"""User-authorized prompt contract; base-generated closing markers are recorded."""
import gzip
import hashlib
import json
import re
from pathlib import Path

BASE = Path('/data/models/Qwen3.5-2B')
EMPTY_THINK = re.compile(r'<think>\s*</think>\s*$')


def check_prompt_contract(tokenizer):
    from patr_core import chat_ids, turn_suffix, _TURN_SUFFIX
    calls = []
    class TrackedTokenizer:
        def __getattr__(self, name): return getattr(tokenizer, name)
        def apply_chat_template(self, *args, **kwargs):
            calls.append(kwargs.copy())
            return tokenizer.apply_chat_template(*args, **kwargs)
    tracked = TrackedTokenizer()
    ids = chat_ids(tracked, [{'role':'user', 'content':'Move to the goal.'}])
    first = tokenizer.decode(ids, skip_special_tokens=False)
    tail = turn_suffix(tracked, 'Next observation.')
    _TURN_SUFFIX.pop(id(tracked), None)
    assert calls and all(c.get('enable_thinking') is False for c in calls)
    assert EMPTY_THINK.search(first) and EMPTY_THINK.search(tail), 'Missing empty think block in prompt'
    return dict(passed=True, enable_thinking=False, initial_empty_think=True, subsequent_empty_think=True,
                initial_prompt_sha256=hashlib.sha256(first.encode()).hexdigest(),
                subsequent_prompt_sha256=hashlib.sha256(tail.encode()).hexdigest())


def generated_policy(count, source_model):
    base = Path(source_model).resolve() == BASE.resolve()
    return dict(passed=count == 0 or base, generated_thinking_markers=count,
                model=str(source_model), original_base_model=base,
                policy='record_only_for_original_base; SFT_or_unknown_markers_require_manual')


def inspect_smoke(out, tokenizer):
    out = Path(out)
    cfg = json.loads((out/'config.json').read_text())
    prompt = check_prompt_contract(tokenizer)
    versions=[json.loads(line) for line in (out/'source_versions.jsonl').read_text().splitlines()]
    metrics=[json.loads(line) for line in (out/'train_metrics.jsonl').read_text().splitlines()]
    # A base path label alone cannot establish provenance after checkpoint recovery.
    pristine_base=bool(versions) and all(v.get('resume') is False for v in versions) and len(metrics)==1 and metrics[0]['step']==1 and metrics[0].get('resumed_from_step',0)==0
    marker_ids = {tokenizer.convert_tokens_to_ids(t) for t in ['<think>', '</think>']}
    raw_path=out/'ucf_logs/raw_leaves_0001.jsonl.gz' if cfg.get('ucf_corrections') else out/'rollouts_0001.jsonl.gz'
    # Part-B pseudo-row masks start at X and are not generation boundaries.
    # Validate actual leaves so copied reasoning/fence prefixes remain visible.
    with gzip.open(raw_path, 'rt') as f: rows = [json.loads(l) for l in f]
    assert rows
    count = 0
    for row in rows:
        assert len(row['ids']) == len(row['mask'])
        # Every generated segment must be preceded by the empty prompt-side block.
        for i, (token, mask) in enumerate(zip(row['ids'], row['mask'])):
            if mask and (i == 0 or not row['mask'][i-1]):
                assert EMPTY_THINK.search(tokenizer.decode(row['ids'][:i], skip_special_tokens=False))
            if mask and token in marker_ids: count += 1
    policy = generated_policy(count, cfg['model'])
    if policy['original_base_model'] and not pristine_base:
        policy['passed']=False
        policy['reason']='Base initialization provenance not proven; keep manual'
    policy['fresh_preupdate_base_provenance']=pristine_base if policy['original_base_model'] else None
    return dict(**policy, prompt_contract=prompt,
                config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
                rollout_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                rollout_source=str(raw_path),pseudo_rows_excluded=bool(cfg.get('ucf_corrections')))
