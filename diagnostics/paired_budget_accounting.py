"""Compare completed exploratory boundaries. Reads fixed validation and metric prefixes only."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def outcomes(run, step):
    summary_path = run / f'validation_{step:04}.json'
    summary = load(summary_path)
    raw = summary_path.with_suffix('.jsonl')
    if raw.exists():
        content = raw.read_bytes()
    else:
        content = gzip.decompress(raw.with_suffix('.jsonl.gz').read_bytes())
    rows = collections.defaultdict(dict)
    for line in content.splitlines():
        record = json.loads(line)
        uid, sample = record['task']['uid'], str(record['lineage'])
        assert sample in ('0', '1', '2', '3') and sample not in rows[uid]
        assert record['reward'] in (0, 1) and record['task']['split'] == 'validation'
        assert (record['terminal'] == 'success') == bool(record['reward'])
        rows[uid][sample] = record
    assert len(rows) == summary['n_tasks'] == 100 and summary['n_trajectories'] == 400
    assert all(len(group) == 4 for group in rows.values())
    flat = [r for group in rows.values() for r in group.values()]
    assert collections.Counter(r['terminal'] for r in flat) == summary['terminal_counts']
    assert math.isclose(sum(r['reward'] for r in flat) / 4, summary['success_rate_pct'])
    assert sum(any(r['reward'] for r in group.values()) for group in rows.values()) == summary['pass_at_4_pct']
    assert math.isclose(sum(r['response_tokens'] for r in flat) / len(flat), summary['response_length'])
    return summary, rows, hashlib.sha256(content).hexdigest()


def costs(run, step, receipt):
    base = receipt['initial_step']
    records, raw_hash = [], hashlib.sha256()
    with (run / 'train_metrics.jsonl').open('rb') as handle:
        for number in range(base + 1, step + 1):
            line = next(handle)
            record = json.loads(line)
            assert number == record['step']
            raw_hash.update(line)
            records.append(record)
    assert raw_hash.hexdigest() == receipt['metrics_sha256']
    seconds = sum(r['step_seconds'] for r in records)
    diagnostic = sum(r.get('ucf_component_diagnostic_seconds', 0.) for r in records)
    assert 0 <= diagnostic < seconds
    return dict(updates=len(records), step_seconds=seconds, rollout_seconds=sum(r['rollout_seconds'] for r in records),
                component_diagnostic_seconds=diagnostic, seconds_excluding_component_diagnostic=seconds-diagnostic,
                generated_tokens=sum(r['generated_tokens'] for r in records), input_tokens=sum(r['input_tokens'] for r in records),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in records), metrics_prefix_sha256=raw_hash.hexdigest(),
                finite_updates=all(math.isfinite(r[k]) for r in records for k in
                                   ('loss', 'kl', 'grad_norm', 'max_parameter_change')),
                nonzero_parameter_updates=sum(r['max_parameter_change'] > 0 for r in records))


def compare(left, right, step):
    receipts = [load(run / f'stage_{step:04}.json') for run in (left, right)]
    assert all(r['passed'] and not r['final_test_used'] and r['steps'] == step for r in receipts)
    assert receipts[0]['initial_step'] == receipts[1]['initial_step']
    arms = [outcomes(run, step) for run in (left, right)]
    for run, receipt in zip((left, right), receipts):
        digest = hashlib.sha256((run / f'validation_{step:04}.json').read_bytes()).hexdigest()
        assert digest == receipt['validation_sha256'][str(step)]
    assert arms[0][1].keys() == arms[1][1].keys()
    fields = ('success_pp','pass_at_4_pp','token_limit_pp','mean_retained_tokens','mean_executed_turns')
    differences, paired_pass, partial = [], collections.Counter(), collections.Counter()
    for uid in sorted(arms[0][1]):
        groups = [list(a[1][uid].values()) for a in arms]
        assert all(r['task'] == groups[0][0]['task'] for g in groups for r in g)
        values = []
        for rows in groups:
            values.append([25*sum(r['reward'] for r in rows), 100*int(any(r['reward'] for r in rows)),
                           25*sum(r['terminal'] == 'token_limit' for r in rows),
                           sum(r['response_tokens'] for r in rows)/4, sum(r['turns'] for r in rows)/4])
        differences.append(np.subtract(*values))
        passed = tuple(bool(v[1]) for v in values)
        paired_pass[{(True,True):'both',(False,False):'neither',(True,False):'ucf_only',(False,True):'grpo_only'}[passed]] += 1
        partial['more_ucf' if values[0][0] > values[1][0] else 'more_grpo' if values[0][0] < values[1][0] else 'same'] += 1
    diff = np.asarray(differences)
    rng = np.random.default_rng(20261107)
    sims = np.concatenate([diff[rng.integers(0, len(diff), size=(250, len(diff)))].mean(axis=1) for _ in range(40)])
    ci = np.quantile(sims, [.025,.975], axis=0)
    cost = [costs(run, step, r) for run, r in zip((left, right), receipts)]
    return dict(step=step, initial_step=receipts[0]['initial_step'], task_n=len(diff), replicates=len(sims),
                direction='UCF minus matched GRPO; repeated-look one-training-seed fixed validation',
                difference={f:dict(estimate=float(diff[:,i].mean()), task_bootstrap_95=ci[:,i].tolist())
                            for i, f in enumerate(fields)},
                questions_by_at_least_one_success={k:paired_pass[k] for k in ('both','neither','ucf_only','grpo_only')},
                questions_by_number_of_successes=dict(partial), validation_raw_sha256=dict(zip(('ucf','grpo'),[a[2] for a in arms])),
                training_cost=dict(zip(('ucf','grpo'), cost)), ucf_over_grpo={k:cost[0][k]/cost[1][k] for k in
                    ('step_seconds','seconds_excluding_component_diagnostic','generated_tokens','input_tokens')},
                cost_scope='Training metrics for matched completed updates only; excludes validation, engine initialization and checkpoint IO outside step timing. Physical generated tokens include discarded suffixes. This is not an equal-compute comparison.',
                final_test_used=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ucf', type=Path, required=True); p.add_argument('--grpo', type=Path, required=True)
    p.add_argument('--step', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.ucf, args.grpo, args.step)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(args.output)
