"""Prepare, but do not dispatch, a hash-bound fresh matched numerical protocol."""
import hashlib
import json
from pathlib import Path

r=Path('/home/vipuser/decision-point-ucf-v3-scoped-20261110')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in (r/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
files += [r/'storage/scoped_inference.py',r/'storage/train_scoped.py',r/'control/v3_manager.py']
assert all(p.stat().st_size < 5_000_000 for p in files), 'Unexpected large runtime artifact'
plan=dict(revision='20261110_v3_scoped_native_training_01',
 state='/data/runs/ucf_v3_paired_scoped_20261110',runtime=str(r/'code'),
 python='/data/venvs/qwen35/bin/python',
 locks=['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock'],
 train_entrypoint=str(r/'storage/train_scoped.py'),scoped_helper=str(r/'storage/scoped_inference.py'),
 source_sha256={str(p):sha(p) for p in sorted(files)},entries=[],
 provenance='Fresh raw-model starts for both arms; the earlier default-kernel25-step baseline is retained as engineering evidence, not the matched control.',
 automatic_400_step_promotion=False,final_test_used=False)
for stop in [25,50,100,150,200]:
 for method in ['ucf','grpo']:
  label=f'V3_S_{method}_nosft_scoped_seed42'
  cfg=r/'v3_configs'/f'{label}.json'
  output=Path('/home/vipuser/ucf-v3-runs-20261110' if method=='ucf' else '/data/runs/ucf_v3_scoped_runs')/label
  plan['entries'].append(dict(id=label,kind='train',method=method,stop_step=stop,
   config=str(cfg),config_sha256=sha(cfg),output=str(output),
   requires_gpu_acceptance='/data/runs/ucf_v3_numerical_scoped_20261110/S/acceptance.json',
   requires_scoped_gpu_acceptance='/data/runs/ucf_v3_numerical_scoped_20261110/S'))
p=r/'v3_plan_scoped.json';p.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(plan=str(p),sha256=sha(p),files=len(files),dispatched=False)))
