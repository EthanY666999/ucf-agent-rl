"""Hash an existing exported model without loading or copying its tensors."""
import hashlib
import json
from pathlib import Path


def identity(root):
    root=Path(root);files={}
    paths=sorted(p for p in root.iterdir() if p.is_file() and
        (p.suffix in {'.safetensors','.json','.jinja','.txt','model'}))
    if not any(p.suffix=='.safetensors' for p in paths):raise ValueError('No exported model weights')
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        files[p.name]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
    combined=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return dict(files=files,sha256=combined)


def verify_sft(root):
    root=Path(root)
    status=json.loads((root/'status.json').read_text())
    if status.get('status')!='complete' or status.get('steps',status.get('step'))!=63:
        raise ValueError('SFT is not a completed 63-step run')
    acceptance=json.loads((root/'sft_acceptance.json').read_text())
    if acceptance.get('passed') is not True:raise ValueError('SFT reload evaluation not accepted')
    ev=json.loads((root/'evaluation/status.json').read_text())
    if ev.get('status')!='evaluation_complete':raise ValueError('SFT reload evaluation incomplete')
    current=identity(root/'final_model')
    if current!=acceptance['model_identity']:raise ValueError('SFT model changed after reload evaluation')
    for name,digest in acceptance['evidence_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('SFT evidence changed: '+name)
    return current
