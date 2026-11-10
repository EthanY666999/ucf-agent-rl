"""User-authorized budget diagnostic at a complete optimizer/checkpoint boundary."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/vipuser/decision-point-ucf-v3-capture-candidate-20261110')
STATE=Path('/data/runs/ucf_v3_budget10240_20261110')
SOURCE=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42')
PYTHON='/data/venvs/qwen35/bin/python'
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

STATE.mkdir(parents=True,exist_ok=True)
own=(STATE/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
write(STATE/'status.json',dict(status='waiting_for_checkpoint16_and_gpu_locks',pid=os.getpid(),unix=time.time()))
locks=[]
for path in ['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock']:
    f=open(path,'a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
status=read(SOURCE/'status.json')
assert status['status']=='paused_checkpointed' and status['step']==16,status
assert read(ROOT/'cpu_test_receipt.json')['returncode']==0
assert read(ROOT/'recorded_prefix_receipt.json')['passed']
cp=SOURCE/'checkpoint.pt';digest=sha(cp)
meta=json.loads(subprocess.check_output([PYTHON,'-c',
    'import torch,json,sys;d=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True);print(json.dumps({"step":d["step"],"config":d["config"]}))',str(cp)],
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),text=True))
assert meta['step']==16 and meta['config']['max_response_tokens']==4096
cfg=dict(meta['config'],max_response_tokens=10240,ucf_probability_source='original_sampling_logits')
config=STATE/'admission_config.json';write(config,cfg)
write(STATE/'source_checkpoint.json',dict(path=str(cp),bytes=cp.stat().st_size,sha256=digest,
    step=16,original_config=meta['config'],unchanged_model_optimizer_rng=True,
    source_task_sft=False,protocol_change='4096 -> 10240 generated tokens per entire trajectory',
    planned_pause=True,source_status=status,unix=time.time()))
paths=[ROOT/'check_budget_gpu.py',ROOT/'code/train.py',ROOT/'code/patr_core.py',
       ROOT/'code/ucf/ucf_v3.py',ROOT/'code/ucf/original_logits.py',ROOT/'storage/scoped_inference.py']
write(STATE/'admission_sources.json',{str(p):sha(p) for p in paths})
env=dict(os.environ,UCF_INHERITED_QUEUE_LOCK_FD=str(locks[0].fileno()),
    UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()))
env.pop('CUDA_VISIBLE_DEVICES',None)
cmd=[PYTHON,'-u',str(ROOT/'check_budget_gpu.py'),'--config',str(config),'--output',str(STATE/'admission'),
    '--checkpoint',str(cp),'--checkpoint-step','16','--source-sha256',digest]
if (STATE/'retry_replay.json').exists():
    replay=read(STATE/'retry_replay.json')
    for path,expected in replay['unchanged_model_code_sha256'].items():assert sha(path)==expected
    cmd+=['--paired-replay-dir',replay['paired_replay_dir']]
with (STATE/'admission.log').open('a') as log:
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        pass_fds=tuple(f.fileno() for f in locks))
    write(STATE/'status.json',dict(status='paired_budget_and_capture_admission',pid=p.pid,manager_pid=os.getpid(),unix=time.time()))
    rc=p.wait()
write(STATE/'admission_exit.json',dict(returncode=rc,unix=time.time()))
write(STATE/'status.json',dict(status='admission_passed' if rc==0 else 'admission_failed_evidence_preserved',
    returncode=rc,unix=time.time(),gpu_locks_released_on_exit=True))
