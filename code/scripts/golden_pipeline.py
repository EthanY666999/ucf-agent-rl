"""Fail-fast sequential SFT -> GRPO acceptance -> PaTR pipeline."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from patr_core import atomic_json
OUT=ROOT/'runs/golden_sft'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PYTHON=sys.executable
model=ROOT/'artifacts/Qwen2.5-0.5B-Instruct'
data=ROOT.parent/'golden-trace/sft_messages.jsonl'
sft=OUT/'sft'
cfg=json.loads((ROOT/'configs/frozenlake_05b.json').read_text())
cfg['model']=str(sft/'final_model')
cfg['run_label']='golden_action_sft_1epoch_then_rl_seed42'
config=ROOT/'configs/golden_sft.json'
atomic_json(config,cfg)
os.environ['SFT_TEST_MODEL']=str(model)
# A detached supervisor can adopt the already-running SFT process after reordering.
if '--after-sft' in sys.argv:
    child=int(sys.argv[sys.argv.index('--after-sft')+1])
    atomic_json(OUT/'pipeline_status.json',dict(status='running',stage='waiting_for_existing_sft',pid=os.getpid(),sft_pid=child))
    while Path(f'/proc/{child}').exists():
        if Path(f'/proc/{child}/stat').read_text().split()[2] == 'Z': break
        time.sleep(5)
    assert json.loads((sft/'status.json').read_text())['status']=='complete', 'Existing SFT did not complete'
stages=[
 ('model_verify',[PYTHON,'scripts/prepare_model.py','--verify-only']),
 ('data_verify',[PYTHON,str(ROOT.parent/'golden-trace/verify.py')]),
 ('unit_tests',[PYTHON,'-m','unittest','discover','-s','tests','-v']),
 ('mask_audit',[PYTHON,'sft.py','--model',str(model),'--data',str(data),'--output',str(sft),'--audit-only']),
 ('sft',[PYTHON,'-u','sft.py','--model',str(model),'--data',str(data),'--output',str(sft)]),
 ('grpo_smoke',[PYTHON,'-u','scripts/run.py','--method','grpo','--config',str(config),'--smoke']),
 ('sft_test',[PYTHON,'-u','scripts/evaluate.py','--model',str(sft/'final_model'),'--output',str(OUT/'sft_test.json')]),
 ('grpo',[PYTHON,'-u','scripts/run.py','--method','grpo','--config',str(config)]),
 ('grpo_diagnostics',[PYTHON,'scripts/golden_diagnostics.py','--run',str(ROOT/'runs/formal/grpo_seed42')]),
 ('grpo_acceptance',[PYTHON,'scripts/golden_acceptance.py']),
 ('patr_smoke',[PYTHON,'-u','scripts/run.py','--method','patr','--config',str(config),'--smoke']),
 ('patr',[PYTHON,'-u','scripts/run.py','--method','patr','--config',str(config)]),
 ('patr_diagnostics',[PYTHON,'scripts/golden_diagnostics.py','--run',str(ROOT/'runs/formal/patr_seed42')]),
]
if '--after-sft' in sys.argv:
    stages=stages[5:]
start=time.time()
for name,cmd in stages:
    atomic_json(OUT/'pipeline_status.json',dict(status='running',stage=name,pid=os.getpid(),started_unix=start,command=cmd))
    print(json.dumps(dict(stage=name,command=cmd)),flush=True)
    with (OUT/f'{name}.log').open('x') as log:
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        atomic_json(OUT/'pipeline_status.json',dict(status='failed',stage=name,returncode=result.returncode))
        raise SystemExit(result.returncode)
atomic_json(OUT/'pipeline_status.json',dict(status='complete',elapsed_seconds=time.time()-start))
