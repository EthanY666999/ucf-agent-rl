"""GPU migration validation followed by a separately recorded 400-to-800 extension."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patr_core import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--parent', type=Path, required=True)
a = p.parse_args()
parent = a.parent.resolve()
out = ROOT / 'runs/extension'
out.mkdir(parents=True, exist_ok=True)
lock = (out / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def status(stage, state='running', **kwargs):
    atomic_json(out / 'status.json', dict(stage=stage, status=state, pid=os.getpid(),
                                        updated_unix=time.time(), **kwargs))

def run(stage, args):
    status(stage)
    with (out / (stage + '.log')).open('x') as log:
        subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)

try:
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip(), 'GPU already occupied'
    bridge = out / 'bridge_validation_0400.json'
    run('bridge_validation', ['scripts/evaluate.py', '--model', str(parent / 'runs/formal/patr_seed42/final_model'),
                             '--split', 'validation', '--output', str(bridge)])
    old = json.loads((parent / 'runs/formal/patr_seed42/validation_0400.json').read_text())
    new = json.loads(bridge.read_text())
    delta = {k: new[k] - old[k] for k in ['success_rate_pct', 'pass_at_4_pct']}
    atomic_json(out / 'bridge_comparison.json', dict(parent=old, new_gpu=new, delta_pp=delta,
                max_allowed_absolute_delta_pp=2, purpose='Hardware migration check; not model selection'))
    assert all(abs(d) <= 2 for d in delta.values()), 'Hardware validation drift exceeds 2pp; investigate before training'
    run('patr_extension', ['scripts/run.py', '--method', 'patr', '--config', 'configs/golden_extend800.json',
                           '--resume', '--extend-training-budget'])
    run_status = json.loads((ROOT / 'runs/formal/patr_seed42/status.json').read_text())
    assert run_status['status'] == 'complete' and run_status['step'] == 800, run_status
    run('diagnostics', ['scripts/golden_diagnostics.py', '--run', str(ROOT / 'runs/formal/patr_seed42')])
    status('complete', 'complete')
except Exception as exc:
    status('failure', 'failed', error=repr(exc))
    raise
