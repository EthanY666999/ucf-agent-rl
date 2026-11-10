"""Wait for the existing training pipeline, then run fixed supplemental evaluations."""
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

OUT = ROOT / 'runs/golden_sft'
lock = (OUT / 'post_pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status_path = OUT / 'post_pipeline_status.json'
def status(**kwargs):
    atomic_json(status_path, dict(pid=os.getpid(), updated_unix=time.time(), **kwargs))

try:
    status(status='waiting', stage='existing_training_pipeline')
    pipeline_lock = (OUT / 'pipeline.lock').open('a')
    fcntl.flock(pipeline_lock, fcntl.LOCK_EX)
    assert json.loads((OUT / 'pipeline_status.json').read_text())['status'] == 'complete'
    # A released supervisor lock alone cannot prove GPU workers have exited.
    for attempt in range(30):
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader'], text=True).strip()
        if not processes:
            break
        time.sleep(10)
    else:
        raise RuntimeError('GPU still occupied; refusing overlapping evaluation: ' + processes)
    models = [('patr', ROOT / 'runs/formal/patr_seed42/final_model'),
              ('grpo', ROOT / 'runs/formal/grpo_seed42/final_model'),
              ('sft', OUT / 'sft/final_model')]
    for name, model in models:
        status(status='running', stage=name + '_supplemental500')
        command = [sys.executable, '-u', 'scripts/golden_supplemental.py',
                   '--model', str(model), '--output', str(OUT / 'supplemental500' / name)]
        with (OUT / f'{name}_supplemental500.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    status(status='complete', stage='supplemental500')
except Exception as exc:
    status(status='failed', error=repr(exc))
    raise
