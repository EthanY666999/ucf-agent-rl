"""Evaluate a fixed exported model on the archived five-batch 500-task protocol."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--runtime-config', type=Path, help='Copy only memory/backend settings from this experiment config')
    args = parser.parse_args()
    baseline = ROOT.parent / 'results/supplemental500'
    protocol = json.loads((baseline / 'protocol.json').read_text())
    task_bytes = (baseline / 'tasks.json').read_bytes()
    assert hashlib.sha256(task_bytes).hexdigest() == protocol['task_file_sha256']
    tasks = json.loads(task_bytes)
    assert len(tasks) == 500 and len({t['uid'] for t in tasks}) == 500
    for i in range(5):
        rows = [json.loads(line) for line in (baseline / f'patr/batch{i+1}.jsonl').open()]
        assert len(rows) == 400
        assert {r['task']['uid'] for r in rows} == {t['uid'] for t in tasks[i*100:(i+1)*100]}
    if args.audit_only:
        print('Task hash and all five baseline batch partitions verified.')
        return
    model = args.model.resolve()
    from collaborator_support import validate_model_dir, evaluation_runtime
    validate_model_dir(model)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from train import build_models, evaluate, RUNTIME_KEYS
    from patr_core import atomic_json
    cfg = evaluation_runtime(protocol['evaluation_config'], args.runtime_config, ROOT, RUNTIME_KEYS)
    cfg['model'] = str(model)
    identity = dict(model=str(model), task_sha256=protocol['task_file_sha256'],
                    config=cfg, seed_per_batch=protocol['evaluation_seed'],
                    sources={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                             for f in ['train.py', 'patr_core.py', 'scripts/golden_supplemental.py']})
    atomic_json(output / 'identity.json', identity)
    actor, ref, engine, tokenizer = build_models(cfg, protocol['model_seed'])
    batches = []
    for i in range(5):
        batches.append(evaluate(engine, tokenizer, cfg, tasks[i*100:(i+1)*100],
                                output / f'batch{i+1}.json', protocol['evaluation_seed'], 'supplemental500'))
        atomic_json(output / 'status.json', dict(status='running', completed_batches=i+1, total_batches=5))
    records = [json.loads(line) for i in range(5)
               for line in (output / f'batch{i+1}.jsonl').open()]
    assert len(records) == 2000
    grouped = {t['uid']: [] for t in tasks}
    for r in records:
        grouped[r['task']['uid']].append(r)
    assert all(len(group) == 4 for group in grouped.values())
    actions = [t['action'] for r in records for t in r['trace'] if t['action'] is not None]
    summary = dict(split='supplemental500', n_tasks=500, n_trajectories=2000,
                   success_rate_pct=100*sum(r['reward'] > 0 for r in records)/2000,
                   pass_at_4_pct=100*sum(any(r['reward'] > 0 for r in g) for g in grouped.values())/500,
                   invalid_action_fraction=sum(a == 0 for a in actions)/len(actions) if actions else None,
                   missing_action_outputs=sum(t['action'] is None for r in records for t in r['trace']),
                   turns=sum(r['turns'] for r in records)/2000,
                   response_length=sum(r['response_tokens'] for r in records)/2000,
                   terminal_counts=dict(Counter(r['terminal'] for r in records)),
                   generated_tokens=sum(b['generated_tokens'] for b in batches),
                   seconds=sum(b['seconds'] for b in batches))
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'status.json', dict(status='complete', completed_batches=5, total_batches=5))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
