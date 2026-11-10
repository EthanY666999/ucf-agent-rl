"""No live intervention: wait for step25, test observer, then run paired GRPO25."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/home/vipuser/decision-point-ucf-v3-capture-candidate-20261110')
state=Path('/data/runs/ucf_v3_capture_boundary_20261110')
previous=Path('/data/runs/ucf_v3_paired_fastscan_20261110')
ucf=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42')
python='/data/venvs/qwen35/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state.mkdir(parents=True,exist_ok=True)
own=(state/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
write(state/'status.json',dict(status='waiting_for_completed_stage25',pid=os.getpid(),unix=time.time()))
locks=[]
for path in ['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock']:
    f=open(path,'a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
env=dict(os.environ,UCF_INHERITED_QUEUE_LOCK_FD=str(locks[0].fileno()),
         UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()))
env.pop('CUDA_VISIBLE_DEVICES',None)
manifest=read(root/'candidate_manifest.json')
assert all(sha(p)==digest for p,digest in manifest['sha256'].items())
assert read(root/'cpu_test_receipt.json')['returncode']==0
assert read(root/'recorded_prefix_receipt.json')['passed']
boundary=read(previous/'boundary_held.json')
assert not boundary['training_interrupted']
stage_ok=(ucf/'stage_0025.json').exists() and read(ucf/'stage_0025.json')['passed']
write(state/'handoff.json',dict(stage25_passed=stage_ok,source_manifest_sha256=sha(root/'candidate_manifest.json'),
    training_interrupted=False,unix=time.time(),automatic_learning_claim=False))
if stage_ok:
    for tag in ['S','F']:
        if tag=='F' and not (state/'S/acceptance.json').exists():break
        cmd=[python,'-u',str(root/'check_gpu.py'),'--config',str(root/f'configs/{tag}.json'),
             '--output',str(state/tag)]
        with (state/f'{tag}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=tuple(f.fileno() for f in locks))
            write(state/'status.json',dict(status='gpu_probability_admission',job=tag,pid=p.pid,unix=time.time()))
            timed_out=False
            try:rc=p.wait(timeout=2700)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:rc=p.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
            write(state/f'{tag}_exit.json',dict(returncode=rc,timed_out=timed_out,unix=time.time()))
        if rc:break

# The unchanged matched baseline is useful even if the experimental observer
# is rejected. Its manager retains all source/config/dataset and disk guards.
plan=read(root/'baseline_plan.json')
assert sha(root/'baseline_plan.json')==manifest['baseline_plan_sha256']
cmd=[python,'-u',plan['manager_script'],str(root/'baseline_plan.json')]
with (state/'baseline_manager.log').open('a') as log:
    p=subprocess.Popen(cmd,cwd=Path(plan['runtime']).parent,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        pass_fds=tuple(f.fileno() for f in locks))
    write(state/'status.json',dict(status='matched_grpo25',manager_pid=p.pid,unix=time.time()))
    rc=p.wait()
write(state/'status.json',dict(status='boundary_plan_ended',baseline_manager_returncode=rc,
    diagnostics_admitted=all((state/t/'acceptance.json').exists() for t in ['S','F']),
    automatic_400_step_promotion=False,unix=time.time()))
