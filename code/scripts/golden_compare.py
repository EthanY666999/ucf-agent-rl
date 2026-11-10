"""Analyze completed fixed test sets without modifying training or evaluation."""
import collections
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patr_core import make_env, distance, atomic_json

def read(paths):
    return [json.loads(line) for p in paths for line in p.open()]

def summarize(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['task']['uid']].append(r)
    assert all(len(g) == 4 for g in grouped.values())
    actions = [t['action'] for r in rows for t in r['trace'] if t['action'] is not None]
    first = [r['trace'][0]['action'] if r['trace'] else None for r in rows]
    successes = [r for r in rows if r['reward'] > 0]
    shortest = {uid: distance(make_env(g[0]['task'], dict(is_slippery=False, max_turns=10)))
                for uid, g in grouped.items()}
    excess = [r['turns'] - shortest[r['task']['uid']] for r in successes]
    invalid_on_success = [sum(t['action'] == 0 for t in r['trace']) for r in successes]
    assert all(d >= 0 for d in excess)
    return dict(n_tasks=len(grouped), success_rate_pct=100*len(successes)/len(rows),
                pass_at_4_pct=100*sum(any(r['reward'] > 0 for r in g) for g in grouped.values())/len(grouped),
                first_turn_legal_pct=100*sum(a in (1,2,3,4) for a in first)/len(first),
                invalid_action_pct=100*sum(a == 0 for a in actions)/len(actions),
                missing_outputs=sum(t['action'] is None for r in rows for t in r['trace']),
                successful_mean_turns=statistics.mean(r['turns'] for r in successes),
                successful_mean_excess_steps=statistics.mean(excess),
                successful_mean_invalid_actions=statistics.mean(invalid_on_success),
                successful_mean_excess_excluding_invalid=statistics.mean(e-i for e,i in zip(excess, invalid_on_success))), grouped

def paired(new, old):
    assert set(new) == set(old)
    keys = sorted(new)
    for uid in keys:
        assert new[uid][0]['task'] == old[uid][0]['task']
    def scores(group):
        return [sum(r['reward'] > 0 for r in group)/4, float(any(r['reward'] > 0 for r in group))]
    delta = np.array([np.subtract(scores(new[k]), scores(old[k])) for k in keys])
    rng = np.random.default_rng(42)
    samples = np.concatenate([delta[rng.integers(0, len(keys), (1000, len(keys)))].mean(1) for _ in range(50)]) * 100
    return dict(delta_success_pp=100*float(delta[:,0].mean()), delta_pass4_pp=100*float(delta[:,1].mean()),
                paired_task_bootstrap_95pct=np.quantile(samples,[.025,.975],axis=0).T.tolist(), replicates=50000,
                caveat='Single training seed; task bootstrap does not estimate training-seed or duplicate-map cluster uncertainty.')

report = {}
for split in ['main100', 'supplemental500']:
    baseline = ROOT.parent / 'results' / split
    if split == 'main100':
        paths = {'old_patr': [baseline/'runs/patr_seed42/final_test.jsonl'],
                 'sft_patr': [ROOT/'runs/formal/patr_seed42/final_test.jsonl'],
                 'sft_grpo': [ROOT/'runs/formal/grpo_seed42/final_test.jsonl'],
                 'sft': [ROOT/'runs/golden_sft/sft_test.jsonl']}
    else:
        paths = {'old_patr': [baseline/f'patr/batch{i}.jsonl' for i in range(1,6)]}
        paths.update({label: [ROOT/f'runs/golden_sft/supplemental500/{method}/batch{i}.jsonl' for i in range(1,6)]
                      for label,method in [('sft_patr','patr'),('sft_grpo','grpo'),('sft','sft')]})
    grouped = {}
    result = {}
    for label, files in paths.items():
        result[label], grouped[label] = summarize(read(files))
    result['sft_patr_minus_old_patr'] = paired(grouped['sft_patr'], grouped['old_patr'])
    result['sft_patr_minus_sft_grpo'] = paired(grouped['sft_patr'], grouped['sft_grpo'])
    result['path_note'] = 'Path lengths/excess condition on each model succeeding; success sets differ.'
    report[split] = result
report['training'] = {}
for method in ['grpo','patr']:
    p = ROOT/f'runs/formal/{method}_seed42'
    rows = read([p/'train_metrics.jsonl'])
    assert [r['step'] for r in rows] == list(range(1,401))
    assert all(math.isfinite(r[k]) for r in rows for k in ['loss','grad_norm','max_parameter_change'])
    diag = json.loads((p/'golden_diagnostics.json').read_text())
    report['training'][method] = dict(updated_steps=sum(r['max_parameter_change'] > 0 for r in rows),
        generated_tokens=sum(r['generated_tokens'] for r in rows), environment_actions=sum(r['executed_actions'] for r in rows),
        step_seconds=sum(r['step_seconds'] for r in rows), elapsed_seconds=rows[-1]['elapsed_seconds'],
        mean_mixed_reward_group_fraction=statistics.mean(r['mixed_reward_group_fraction'] for r in diag),
        last40_mixed_reward_group_fraction=statistics.mean(r['mixed_reward_group_fraction'] for r in diag[-40:]))
report['sft_extra_seconds'] = 109.065
report['training_note'] = 'All completed 400-step runs and test evaluations used A100. Sampling budgets differ; equal steps do not mean equal compute. Mixed groups use final retained branches, with shared prefixes repeated.'
atomic_json(ROOT/'runs/golden_sft/completed_comparison.json', report)
print(json.dumps(report, indent=2))
