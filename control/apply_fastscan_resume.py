"""Apply a CPU-equivalent decoder optimization at a saved update boundary."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

r=Path('/home/vipuser/decision-point-ucf-v3-scoped-20261110')
c=Path('/home/vipuser/ucf-prefix-cpu-candidate-20261110')
out=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42')
old=read(r/'v3_plan_scoped.json')
state=Path('/data/runs/ucf_v3_paired_fastscan_20261110')
assert not (state/'launch.json').exists(), 'Already dispatched'
locks=[]
for name in old['locks']:
    f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
status=read(out/'status.json');assert status['status']=='paused_checkpointed' and status['step']==1
for path,h in old['source_sha256'].items():assert sha(Path(path))==h,path
proof=read(c/'equivalence.json');assert proof['passed'] and proof['checked']==180
assert proof['prefix_causality_no_suffix_access'] and proof['all_single_token_backticks_covered']
state.mkdir(parents=True,exist_ok=True)
write(state/'original_plan.json',old)
write(state/'planned_pause_receipt.json',dict(step=1,status=status,checkpoint_bytes=(out/'checkpoint.pt').stat().st_size,
    reason='CPU-equivalent prefix scan optimization, not failed learning',final_test_used=False,unix=time.time()))
for relative,h in proof['candidate_sha256'].items():
    src=c/relative;dest=r/'code'/relative;assert sha(src)==h
    if dest.exists():
        backup=state/'before_source'/relative;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
    temp=dest.with_name(dest.name+'.fastscan.tmp');shutil.copy2(src,temp);temp.replace(dest)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
    GIT_DIR='/home/vipuser/decision-point-ucf-v3-20261110/.git',
    PYTHONPATH='/home/vipuser/decision-point-ucf-v3-20261110/.test-deps:'+str(r/'code')+':'+str(r/'code/ucf'))
tests=['control/test_component_cadence.py','control/test_v3_manager.py','code/ucf/tests/test_v3_sampling.py',
 'code/ucf/tests/test_v3_collector.py','code/ucf/tests/test_ucf.py','code/ucf/tests/test_first_fence.py',
 'code/ucf/tests/test_causal_complement.py','code/tests/test_response_protocol.py','code/tests/test_hard_envs.py',
 'code/tests/test_ucf_recovery.py','code/tests/test_pilot_stop_boundary.py','code/tests/test_validation_recovery.py']
with (state/'cpu_tests.log').open('w') as log:
    p=subprocess.run([old['python'],'-m','pytest','-q',*tests],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,'CPU regression failed; evidence preserved and no GPU launch'
plan=dict(old,state=str(state),revision='20261110_v3_scoped_native_training_fastscan_02')
plan['source_sha256']={path:sha(Path(path)) for path in old['source_sha256']}
plan['source_sha256'][str(r/'code/token_boundaries.py')]=sha(r/'code/token_boundaries.py')
plan['performance_equivalence_sha256']=sha(c/'equivalence.json')
path=r/'v3_plan_scoped_fastscan.json';write(path,plan);write(state/'plan.json',plan)
write(state/'admission.json',dict(passed=True,plan_sha256=sha(path),cpu_log_sha256=sha(state/'cpu_tests.log'),
    equivalence_sha256=sha(c/'equivalence.json'),algorithm_changed=False,checkpoint_resumed_from=1,unix=time.time()))
env=dict(os.environ,UCF_INHERITED_QUEUE_LOCK_FD=str(locks[0].fileno()),UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()))
env.pop('CUDA_VISIBLE_DEVICES',None)
with (state/'manager.log').open('a') as log:
    proc=subprocess.Popen([old['python'],'-u',str(r/'control/v3_manager.py'),str(path)],cwd=r,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        pass_fds=tuple(f.fileno() for f in locks))
record=dict(pid=proc.pid,unix=time.time(),plan_sha256=sha(path),resumed_checkpoint_step=1,algorithm_changed=False)
write(state/'launch.json',record);print(json.dumps(record))
