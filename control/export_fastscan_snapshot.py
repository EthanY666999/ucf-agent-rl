"""Read-only evidence snapshot; no models or optimizer states are exported."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

run = Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42')
state = Path('/data/runs/ucf_v3_paired_fastscan_20261110')
files = []
for name in ['config.json', 'launch_config.json', 'hardware.json', 'prompt_contract.json',
             'numerical_runtime.json', 'source_versions.jsonl', 'validation_0000.json',
             'validation_0000.jsonl']:
    p = run / name
    if p.is_file():
        files.append((p, 'results/scoped_ucf_step1/' + name))
for name in ['budget_0001.json', 'select_0001.jsonl', 'controller_0001.json']:
    files.append((run / 'ucf_logs' / name, 'results/scoped_ucf_step1/ucf_logs/' + name))
for name in ['cpu_tests.log', 'admission.json', 'launch.json', 'plan.json', 'planned_pause_receipt.json']:
    files.append((state / name, 'reports/fastscan_resume/' + name))

metrics = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines() if line.strip()]
first = [r for r in metrics if r['step'] == 1]
assert len(first) == 1
manifest = []
archive = state / 'step1_light_evidence.tar.gz'
total = 0

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

inventory = []
for p in [*sorted(run.glob('rollouts_0001*')), run / 'ucf_logs/raw_leaves_0001.jsonl.gz']:
    if p.is_file():
        inventory.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p),
                              retained_remote=True, uploaded_in_this_snapshot=False))

with tarfile.open(archive, 'w:gz') as tf:
    def add(name, b):
        entry = tarfile.TarInfo(name)
        entry.size = len(b)
        tf.addfile(entry, io.BytesIO(b))

    for p, name in files:
        assert p.suffix not in {'.pt', '.safetensors', '.bin'}
        raw = p.read_bytes()
        assert len(raw) < 20_000_000, p
        total += len(raw)
        assert total < 40_000_000
        content = gzip.compress(raw, mtime=0) if name.endswith('.jsonl') else raw
        if name.endswith('.jsonl'):
            name += '.gz'
        add(name, content)
        manifest.append(dict(path=str(p), archive_path=name, bytes=len(raw),
                             source_sha256=hashlib.sha256(raw).hexdigest(),
                             archive_sha256=hashlib.sha256(content).hexdigest()))
    add('results/scoped_ucf_step1/train_metrics.jsonl',
        (json.dumps(first[0]) + '\n').encode())
    summary = dict(unix=time.time(), files=manifest, retained_rollouts=inventory,
                   no_models_or_checkpoints_transferred=True,
                   disks={m: dict(free_bytes=shutil.disk_usage(m).free) for m in ['/home/vipuser', '/data']})
    add('reports/fastscan_resume/evidence_manifest.json', (json.dumps(summary, indent=2) + '\n').encode())
print(json.dumps(dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
                     files=len(files), models_transferred=False)))
