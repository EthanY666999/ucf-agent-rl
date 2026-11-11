"""Archive the two verified stage-100 continuations without launching training."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ROOT = Path('/data/runs/ucf_v3_budget10240_20261110')
ARMS = ('ucf', 'grpo')
TOTAL_FIELDS = ('generated_tokens', 'input_tokens', 'step_seconds', 'rollout_seconds')


def read(path):
    return json.loads(path.read_bytes())


def write(path, record):
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_export(report):
    manifest = read(report)
    assert manifest['passed'] and not manifest['final_test_used']
    for name, expected in manifest['files'].items():
        p = report.parent / name
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], p
    return manifest


def segment_cost(run):
    """Verify both historical metric prefixes, then account only for new updates."""
    lines = (run / 'train_metrics.jsonl').read_bytes().splitlines(keepends=True)
    prior, current = [read(run / f'stage_{step:04d}.json') for step in (50, 100)]
    start = prior['initial_step']
    for step, receipt in ((50, prior), (100, current)):
        assert receipt['passed'] and not receipt['final_test_used'] and receipt['steps'] == step
        assert receipt['initial_step'] == start
        assert receipt['metrics_sha256'] == hashlib.sha256(b''.join(lines[:step - start])).hexdigest()
    rows = [json.loads(line) for line in lines]
    assert [r['step'] for r in rows] == list(range(start + 1, 101))
    updates = rows[50 - start:]
    assert len(updates) == 50
    for r in updates:
        assert r['max_parameter_change'] > 0 and all(math.isfinite(r[k]) for k in
            ('loss', 'kl', 'grad_norm', 'max_parameter_change', *TOTAL_FIELDS))
    sums = {k: sum(r[k] for r in updates) for k in TOTAL_FIELDS}
    timing = sum(r.get('ucf_component_diagnostic_seconds', 0.) for r in updates)
    return dict(first_step=51, last_step=100, updates=len(updates), **sums,
                seconds_excluding_component_diagnostic=sums['step_seconds'] - timing,
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in updates),
                metrics50_sha256=prior['metrics_sha256'], metrics100_sha256=current['metrics_sha256'])


def archive_arm(root, arm, run):
    from paired_budget_accounting import outcomes
    from export_budget_boundary import package

    report_dir = root / 'reports'
    cost = segment_cost(run)
    fixed_tasks = None
    summaries = {}
    receipt = read(run / 'stage_0100.json')
    for step in (50, 75, 100):
        summary, groups, raw_hash = outcomes(run, step)
        tasks = {uid: group['0']['task'] for uid, group in groups.items()}
        assert fixed_tasks is None or fixed_tasks == tasks
        fixed_tasks = tasks
        assert sha(run / f'validation_{step:04d}.json') == receipt['validation_sha256'][str(step)]
        summaries[str(step)] = dict(n_tasks=len(groups), n_trajectories=summary['n_trajectories'],
                                   successful_trajectories=sum(r['reward'] for group in groups.values() for r in group.values()),
                                   success_rate_pct=summary['success_rate_pct'], pass_at_4_pct=summary['pass_at_4_pct'],
                                   raw_sha256=raw_hash, summary_sha256=receipt['validation_sha256'][str(step)])
    archive = report_dir / f'{arm}_boundary_0100.tar.gz'
    result = package(run, 100, archive)
    record = dict(arm=arm, run=str(run), passed=True, final_test_used=False, fixed_validation=summaries,
                  new_training_cost=cost, archive=dict(path=str(archive), **result))
    write(report_dir / f'{arm}_continuation_0100.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arm', choices=ARMS, help='Archive one arm immediately after its validated stage-100 receipt.')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report_dir = root / 'reports'
    plan = read(root / 'plan.json')
    runs = {e['method']: Path(e['output']) for e in plan['entries'] if e['stop_step'] == 100}
    assert set(runs) == set(ARMS)
    with (report_dir / 'archive_continuation_0100.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = report_dir / 'paired_export_0050.json'
        previous_hash = sha(previous)
        assert verify_export(previous)['step'] == 50
        if args.arm:
            record = archive_arm(root, args.arm, runs[args.arm])
            print(json.dumps(record))
            if args.arm == "ucf":
                storage = root / "paired100" / "storage"
                if (storage / "models_released.json").exists():
                    # UCF has made its last atomic save. Rebalance only
                    # verified idle archives before the data-volume writer runs.
                    subprocess.run([sys.executable, str(storage / "paired100_storage.py"),
                                    "reserve", "grpo", "--target", "35", "--idle-floor", "7"], check=True)
            return
        records = {arm: archive_arm(root, arm, runs[arm]) for arm in ARMS}
        paired = report_dir / 'paired_export_0100.json'
        if not paired.exists():
            subprocess.run(['bash', str(root / 'review_budget_pair.sh'), str(root), '100'], check=True)
        manifest = verify_export(paired)
        assert manifest['step'] == 100 and sha(previous) == previous_hash
        for arm, record in records.items():
            assert record['archive']['sha256'] == manifest['files'][f'{arm}_boundary_0100.tar.gz']['sha256']
        accounting = read(report_dir / 'paired_accounting_0100.json')
        historical = read(report_dir / 'paired_accounting_0050.json')
        for arm, record in records.items():
            for name in TOTAL_FIELDS:
                change = accounting['training_cost'][arm][name] - historical['training_cost'][arm][name]
                assert math.isclose(record['new_training_cost'][name], change, rel_tol=1e-10, abs_tol=1e-6)
        result = dict(passed=True, final_test_used=False, historical_export_sha256=previous_hash,
                      paired_export_sha256=sha(paired), arms=records,
                      paired_validation=accounting['difference'], cost_scope=accounting['cost_scope'])
        write(report_dir / 'paired_continuation_0100.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
