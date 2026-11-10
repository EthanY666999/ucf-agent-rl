"""Verify the first original rollouts of the two shared-checkpoint continuations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def originals(run, step):
    rows = {}
    with gzip.open(run / f'rollouts_{step:04d}.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            if row['lineage'] in ('0', '1', '2', '3'):
                key = (row['task']['uid'], row['lineage'])
                assert key not in rows
                rows[key] = row
    return rows


def compare(left, right):
    a, b = (read(run/'initialization.json') for run in (left, right))
    for field in ('kind', 'initial_step', 'source_checkpoint_sha256', 'tensor_optimizer_rng_sha256'):
        assert a[field] == b[field], field
    cfg, cfg2 = (read(run/'config.json') for run in (left, right))
    fields = ('model', 'seed', 'dataset_sha256', 'frozen_maps_sha256',
              'max_turns', 'max_response_tokens', 'response_protocol',
              'initial_branches', 'flat_group_size', 'temperature', 'top_p', 'top_k',
              'train_batch_size', 'vllm_batch_invariant_scoped')
    for field in fields:
        assert cfg.get(field) == cfg2.get(field), field
    step = a['initial_step'] + 1
    first, second = originals(left, step), originals(right, step)
    assert len(first) == cfg['train_batch_size'] * cfg['initial_branches']
    assert first.keys() == second.keys()
    differences = []
    maximum_logprob_delta = 0.
    for key in sorted(first):
        x, y = first[key], second[key]
        different = [field for field in ('task', 'ids', 'mask', 'reward', 'terminal') if x[field] != y[field]]
        for field in ('action', 'anchor_before', 'anchor_after'):
            if [r.get(field) for r in x['trace']] != [r.get(field) for r in y['trace']]:
                different.append('trace.' + field)
        if len(x['old_logprobs']) != len(y['old_logprobs']):
            different.append('old_logprobs.length')
        else:
            delta = max((abs(u-v) for u,v in zip(x['old_logprobs'],y['old_logprobs'])), default=0.)
            maximum_logprob_delta = max(maximum_logprob_delta, delta)
            if x['old_logprobs'] != y['old_logprobs']:
                different.append('old_logprobs')
        if different:
            differences.append(dict(task_uid=key[0], lineage=key[1], fields=different))
    return dict(step=step, left=str(left), right=str(right), originals=len(first),
                passed=not differences, exact_originals_equal=not differences,
                maximum_old_logprob_delta=maximum_logprob_delta, differences=differences,
                shared_checkpoint_sha256=a['source_checkpoint_sha256'],
                audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Only the first new update shares model parameters. Later original samples may differ because the methods update differently. Equality is a sampler/initialization check, not a learning result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('differences','left','right')}))
