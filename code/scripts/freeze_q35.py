"""Explicitly freeze a reviewed port; never modifies the historical 0.5B manifest."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=ROOT/'SOURCE_MANIFEST.json';before=h(original)
old=json.loads(original.read_text());files={n:h(ROOT/n) for n in old['files']}
for p in ROOT.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
 rel=str(p.relative_to(ROOT))
 if p.suffix=='.py' or rel.startswith('ucf/') or rel.startswith('eval_sets/') or rel in ['configs/frozenlake_q35_2b.json','configs/external_baselines.json']:
  files[rel]=h(p)
manifest=dict(model_family='Qwen3.5',model='/data/models/Qwen3.5-2B',source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),historical_manifest_sha256=before,files=files)
(ROOT/'SOURCE_MANIFEST_q35.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'artifacts/frozen_q35.json').write_text(json.dumps(dict(model_family='Qwen3.5',source_root=str(ROOT),source_sha256=files,beta=.25,phase1_selection=False,calibration='per_batch_target2_capacity_limited_as_approved_20261107'),indent=2)+'\n')
assert h(original)==before
print(json.dumps(dict(files=len(files),historical_manifest_unchanged=True)))
