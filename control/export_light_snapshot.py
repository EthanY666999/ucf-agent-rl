"""Export small evidence only. Models/checkpoints and training rollouts stay remote."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

old=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_grpo_nosft_seed42')
diag=Path('/data/runs/ucf_v3_numerical_scoped_20261110')
state=Path('/data/runs/ucf_v3_paired_scoped_20261110')
assert json.loads((old/'stage_0025.json').read_text())['passed']
assert json.loads((diag/'status.json').read_text())['status']=='diagnostics_finished'
files=[]
for name in ['config.json','hardware.json','prompt_contract.json','stage_0025.json','status.json',
             'train_metrics.jsonl','source_versions.jsonl','validation_0000.json','validation_0000.jsonl',
             'validation_0025.json','validation_0025.jsonl']:
    p=old/name
    if p.is_file():files.append((p,'results/default_numerics_grpo25/'+name))
p=Path('/data/runs/ucf_v3_control_batched_20261110/review_step25.json')
files.append((p,'results/default_numerics_grpo25/review_step25.json'))
for p in diag.rglob('*'):
    if p.is_file() and p.suffix in {'.json','.jsonl','.gz','.log'}:
        files.append((p,'reports/scoped_gpu/'+str(p.relative_to(diag))))
for name in ['launch.json','status.json']:
    if (state/name).is_file():files.append((state/name,'reports/scoped_launch/'+name))
inventory=[]
for p in sorted(old.glob('rollouts_*.jsonl.gz')):
    inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),retained_remote=True,uploaded_in_this_snapshot=False))
cp=old/'checkpoint.pt'
inventory.append(dict(path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp),retained_remote=True,uploaded_in_this_snapshot=False))
archive=state/'light_evidence_20261110.tar.gz'
manifest=[];total=0
with tarfile.open(archive,'w:gz') as tf:
    for p,name in files:
        assert p.suffix not in {'.pt','.safetensors','.bin'}
        raw=p.read_bytes();assert len(raw)<20_000_000,(p,len(raw))
        total+=len(raw);assert total<50_000_000,total
        source_sha=hashlib.sha256(raw).hexdigest()
        content=gzip.compress(raw,mtime=0) if name.endswith('.jsonl') else raw
        if name.endswith('.jsonl'):name+='.gz'
        ti=tarfile.TarInfo(name);ti.size=len(content);tf.addfile(ti,io.BytesIO(content))
        manifest.append(dict(path=str(p),archive_path=name,original_bytes=len(raw),original_sha256=source_sha,
            archived_bytes=len(content),archived_sha256=hashlib.sha256(content).hexdigest()))
    for name,obj in [('reports/scoped_evidence_manifest.json',dict(unix=time.time(),files=manifest)),
                     ('reports/retained_default_grpo25_inventory.json',dict(files=inventory,no_model_or_checkpoint_transferred=True))]:
        b=(json.dumps(obj,indent=2)+'\n').encode();ti=tarfile.TarInfo(name);ti.size=len(b);tf.addfile(ti,io.BytesIO(b))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),models_transferred=False)))
