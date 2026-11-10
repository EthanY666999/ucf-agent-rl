"""Local-only mixed-reward statistics; never transmits or deletes experiment files."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


def summarize(run):
    rows = []
    for f in sorted(Path(run).glob('rollouts_*.jsonl.gz')):
        match = re.fullmatch(r'rollouts_(\d+)\.jsonl\.gz', f.name)
        if not match:
            continue
        groups = defaultdict(list)
        with gzip.open(f, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                groups[r['task']['uid']].append(float(r['reward']) > 0)
        if not groups:
            raise ValueError(f'Empty rollout: {f}')
        n = len(groups)
        success = sum(all(v) for v in groups.values())
        failure = sum(not any(v) for v in groups.values())
        rows.append(dict(step=int(match[1]), tasks=n,
                         traj_per_task=sum(map(len, groups.values())) / n,
                         all_success=success/n, all_fail=failure/n, mixed=(n-success-failure)/n))
    if not rows:
        raise ValueError(f'No completed rollout files in {run}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    rows = summarize(a.run)
    with (a.run / 'signal_stats.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tail = [r for r in rows if 201 <= r['step'] <= 400]
    print(json.dumps(dict(available_steps=len(rows), last_step=rows[-1]['step'],
         steps201_400_present=len(tail), complete_201_400={r['step'] for r in tail} == set(range(201, 401)),
         mixed_201_400_pct=100*sum(r['mixed'] for r in tail)/len(tail) if tail else None,
         note='Descriptive batch mean; incomplete ranges are explicitly flagged. No upload.'), indent=2))


if __name__ == '__main__':
    main()
