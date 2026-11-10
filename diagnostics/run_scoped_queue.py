"""Wait for the committed25-step boundary, then run bounded isolated checks."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/vipuser/decision-point-ucf-v3-20261110')
state=Path('/data/runs/ucf_v3_numerical_scoped_20261110');state.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
atomic(state/'status.json',dict(status='waiting_for_checkpoint25',pid=os.getpid(),unix=time.time()))
locks=[]
for name in ['/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock']:
 f=open(name,'a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
plan=read(r/'v3_plan_batched.json')
boundary=read(Path(plan['state'])/'boundary_held.json')
baseline=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_grpo_nosft_seed42')
assert read(baseline/'status.json')['status']=='paused_checkpointed'
assert read(baseline/'status.json')['step']==25
assert read(baseline/'stage_0025.json')['passed']
assert not boundary['training_interrupted']
for path,digest in plan['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
scripts=[r/'storage/scoped_inference.py',r/'storage/numerical_contract_scoped.py',Path(__file__)]
atomic(state/'manifest.json',dict(parent_pid=os.getpid(),paused_baseline_step=25,
 script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
 installed_files_modified=False,final_test_used=False,baseline_checkpoint_retained=True))
for tag in ['S','F']:
 if tag=='F' and not (state/'S/scope_acceptance.json').is_file():continue
 cmd=['/data/venvs/qwen35/bin/python','-u',str(r/'storage/numerical_contract_scoped.py'),
      '--config',str(r/f'v3_configs/V3_{tag}_ucf_nosft_seed42.json'),'--output',str(state/tag)]
 env=dict(os.environ,UCF_INHERITED_GPU_LOCK_FD=str(locks[1].fileno()));env.pop('CUDA_VISIBLE_DEVICES',None)
 with (state/f'{tag}.log').open('a') as log:
  p=subprocess.Popen(cmd,cwd=r/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(locks[1].fileno(),))
  record=dict(status='running',job=tag,pid=p.pid,command=cmd,unix=time.time());atomic(state/'status.json',record)
  rc=p.wait();record.update(status='ended',returncode=rc,ended=time.time());atomic(state/f'{tag}_exit.json',record)
atomic(state/'status.json',dict(status='diagnostics_finished',unix=time.time(),user_approval_required=False,requires_root_agent_decision=True))
