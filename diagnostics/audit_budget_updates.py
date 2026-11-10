"""CPU-only accounting of completed training samples; never loads model weights."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(run, through):
    config = json.loads((run / 'config.json').read_text())
    origin = json.loads((run / 'initialization.json').read_text())
    metrics_bytes = (run / 'train_metrics.jsonl').read_bytes()
    rows = [json.loads(line) for line in metrics_bytes.splitlines()]
    rows = [row for row in rows if row['step'] <= through]
    assert [r['step'] for r in rows] == list(range(origin['initial_step'] + 1, through + 1))
    result = []
    for row in rows:
        step = row['step']
        path = run / f'rollouts_{step:04d}.jsonl.gz'
        terminals = collections.Counter()
        n = successes = actions = response_tokens = records = 0
        discarded_tokens = 0
        actual_generated_counts = []
        task_rewards = collections.defaultdict(list)
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                trajectory = json.loads(line)
                records += 1
                if str(trajectory['lineage']) not in {'0', '1', '2', '3'}:
                    continue
                n += 1
                successes += trajectory['reward'] > 0
                terminals[trajectory['terminal']] += 1
                # Trace records can include invalid actions, so they are reported
                # separately from the collector's executed-turn metric.
                actions += len(trajectory['trace'])
                retained = sum(trajectory['mask'])
                discarded = sum(turn.get('protocol_cutoff', {}).get('discarded_sampled_tokens', 0)
                                for turn in trajectory['trace'])
                assert retained <= config['max_response_tokens']
                response_tokens += retained
                discarded_tokens += discarded
                actual_generated_counts.append(retained + discarded)
                task_rewards[json.dumps(trajectory['task'], sort_keys=True)].append(trajectory['reward'])
        assert n == config['train_batch_size'] * config['initial_branches']
        assert len(task_rewards) == config['train_batch_size']
        assert all(len(values) == config['initial_branches'] for values in task_rewards.values())
        assert abs(successes / n - row['success_rate']) < 1e-12
        assert abs(response_tokens / n - row['mean_response_tokens']) < 1e-9
        groups = collections.Counter('all_failure' if not any(v > 0 for v in values)
                                     else 'all_success' if all(v > 0 for v in values)
                                     else 'mixed' for values in task_rewards.values())
        groups = {kind: groups[kind] for kind in ('all_failure', 'mixed', 'all_success')}
        if 'initial_group_counts' in row:
            assert groups == row['initial_group_counts']
        differences = row.get('probe_original_logprob_differences', [])
        finite_keys = ['loss', 'kl', 'grad_norm', 'step_seconds', 'max_parameter_change']
        result.append(dict(
            step=step, original_count=n, original_successes=successes,
            original_success_pct=100 * successes / n,
            original_terminals=dict(terminals),
            original_token_limit_pct=100 * terminals['token_limit'] / n,
            original_mean_trace_records=actions / n,
            original_mean_executed_turns=row['mean_turns'],
            original_mean_response_tokens=response_tokens / n,
            original_discarded_suffix_tokens=discarded_tokens,
            original_max_generated_including_discarded=max(actual_generated_counts),
            original_over_budget_including_discarded=sum(value > config['max_response_tokens']
                                                         for value in actual_generated_counts),
            original_group_counts=dict(groups),
            total_saved_records_including_pseudo_rows=records,
            finite_update=all(math.isfinite(row[k]) for k in finite_keys),
            max_abs_capture_logprob_difference=max(map(abs, differences), default=None),
            rollout_file=dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)),
            **{k: row.get(k) for k in [
                'rollout_seconds', 'step_seconds', 'peak_allocated_gib', 'mean_forks',
                'rescue_rate', 'counterfactual_flip_rate', 'kl', 'grad_norm',
                'max_parameter_change', 'ucf_component_diagnostics_computed',
                'original_logits_capture', 'counterfactual_token_partition_parser_disagreement']}
        ))
    return dict(unix=time.time(), run=str(run), through_step=through,
                initialization=origin, config_sha256=digest(run / 'config.json'),
                audit_script_sha256=digest(Path(__file__)),
                observed_metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
                original_filter='lineage exactly 0, 1, 2 or 3; pseudo rows and CF leaves excluded',
                budget_meaning='The configured cap is cumulative retained response tokens per trajectory, excluding prompts/observations. Engine-generated suffixes discarded after the first complete action block are separate computational overhead, included in generated_tokens; they are not executed or trained.',
                caveat='Training samples from different steps are different batches. This audit does not establish fixed-validation improvement or UCF advantage over GRPO.',
                rows=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.through)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(dict(through_step=args.through, original_token_limit_pct=report['rows'][-1]['original_token_limit_pct'], finite=all(r['finite_update'] for r in report['rows']))))
