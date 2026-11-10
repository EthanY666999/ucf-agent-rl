"""Exclusive, bounded diagnostics after the baseline has checkpointed safely."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--state',default='/data/runs/ucf_v3_numerical_20261110');p.add_argument('--skip-replay',action='store_true');a=p.parse_args()
root=Path('/home/vipuser/decision-point-ucf-v3-20261110');state=Path(a.state);state.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
locks=[]
for name in ['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock']:
 f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
prior=Path('/data/runs/ucf_v3_control_20261110');run=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_grpo_nosft_seed42')
s=read(run/'status.json');assert s['status']=='paused_checkpointed' and 0<s['step']<=25
assert (run/'checkpoint.pt').is_file() and (prior/'planned_diagnostic_pause.json').is_file()
assert read(prior/'boundary_held.json')['training_interrupted'] is False
plan=read(root/'v3_plan.json')
for p,h in plan['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
python='/data/venvs/qwen35/bin/python';env=dict(os.environ,UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()));env.pop('CUDA_VISIBLE_DEVICES',None)
jobs=[('probe_replay',[str(root/'storage/probe_replay.py'),'--config',str(root/'v3_configs/V3_S_ucf_nosft_seed42.json'),'--cases',str(root/'storage/probe_cases.json'),'--output',str(state/'probe_replay')]),('batch_invariant_S',[str(root/'storage/numerical_contract.py'),'--batch-invariant','--config',str(root/'v3_configs/V3_S_ucf_nosft_seed42.json'),'--output',str(state/'batch_invariant_S')]),('batch_invariant_F',[str(root/'storage/numerical_contract.py'),'--batch-invariant','--config',str(root/'v3_configs/V3_F_ucf_nosft_seed42.json'),'--output',str(state/'batch_invariant_F')])]
manifest=dict(parent_pid=os.getpid(),paused_baseline_step=s['step'],jobs=jobs,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'storage/probe_replay.py',root/'storage/numerical_contract.py',root/'storage/probe_cases.json']},final_test_used=False)
atomic(state/'manifest.json',manifest)
for name,args in jobs:
 if a.skip_replay and name=='probe_replay':continue
 if name=='batch_invariant_F' and not (state/'batch_invariant_S/acceptance.json').exists():continue
 log=state/(name+'.log')
 with log.open('a') as f:
  child=subprocess.Popen([python,'-u',*args],env=env,cwd=root/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(locks[1].fileno(),))
  record=dict(job=name,pid=child.pid,started=time.time(),status='running');atomic(state/'status.json',record)
  rc=child.wait();record.update(status='ended',returncode=rc,ended=time.time());atomic(state/(name+'_exit.json'),record)
atomic(state/'status.json',dict(status='diagnostics_finished',unix=time.time(),baseline_checkpoint_retained=True,requires_root_agent_decision=True,user_approval_required=False))
