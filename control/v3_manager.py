"""One-GPU diagnostic queue; inherited boundary locks and evidence preservation."""
import fcntl,hashlib,json,math,os,re,shutil,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def identity(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
  return None if s[0]=='Z' else s[19]
 except FileNotFoundError:return None
def cpu_env():return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def checkpoint_step(p,python):
 cmd=[python,'-c','import torch,sys;d=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True);print(int(d["step"]))',str(p)]
 return int(subprocess.check_output(cmd,env=cpu_env(),text=True).strip())

def reconcile(out,step):
 """Archive post-checkpoint evidence before restoring the durable metric prefix."""
 out=Path(out);archive=out/'recovery_archive'/str(time.time_ns())
 records=[];moves=[];rewrites={}
 for name in ('train_metrics.jsonl','branch_events.jsonl'):
  p=out/name
  if not p.exists():continue
  keep=[];later=[]
  for line in p.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:later.append(line);continue
   (keep if r['step']<=step else later).append(line)
  if name=='train_metrics.jsonl':
   records=[json.loads(x)['step'] for x in keep]
   assert records==list(range(1,step+1)), 'Incomplete/duplicate checkpoint metric prefix'
  if later:moves.append(p);rewrites[p]='\n'.join(keep)+('\n' if keep else '')
 if step:assert records,'Missing checkpoint metric prefix'
 pattern=re.compile(r'(?:rollouts|validation)_(\d+)\.(?:json|jsonl|jsonl.gz)$')
 for p in out.iterdir():
  m=pattern.fullmatch(p.name)
  if p.is_file() and m and int(m.group(1))>step:moves.append(p)
 hashes={p.name:sha(p) for p in moves}
 if moves:
  archive.mkdir(parents=True)
  atomic(archive/'manifest.json',dict(checkpoint_step=step,sha256=hashes,reason='preserve uncheckpointed attempt'))
  for p in moves:
   p.rename(archive/p.name)
   if p in rewrites:p.write_text(rewrites[p])
 return dict(checkpoint_step=step,archived_sha256=hashes)

def validate_stage(out,step,python,runtime):
 out=Path(out);s=read(out/'status.json')
 assert s['status']=='paused_checkpointed' and s['step']==step
 assert not (out/'final_test.json').exists(),'Pilot entered final test'
 metrics=[json.loads(x) for x in (out/'train_metrics.jsonl').read_text().splitlines()]
 assert [r['step'] for r in metrics]==list(range(1,step+1))
 assert checkpoint_step(out/'checkpoint.pt',python)==step
 for r in metrics:assert all(math.isfinite(r[k]) for k in ('loss','kl','grad_norm','step_seconds'))
 cfg=read(out/'config.json');assert sha(cfg['dataset_path'])==cfg['dataset_sha256']
 tasks=read(cfg['dataset_path'])['validation']
 sys.path.insert(0,str(Path(runtime)/'scripts'))
 from validation_recovery import validation_complete
 for k in range(0,step+1,25):assert validation_complete(out/f'validation_{k:04d}.json',tasks),f'Incomplete validation{k}'
 return dict(passed=True,steps=step,final_test_used=False,checkpoint_retained=True,
  metrics_sha256=sha(out/'train_metrics.jsonl'),validation_sha256={str(k):sha(out/f'validation_{k:04d}.json') for k in range(0,step+1,25)})

def write_progress(state,entry,child=None):
 out=Path(entry['output']);val=[]
 for p in sorted(out.glob('validation_*.json')):
  try:
   d=read(p);val.append(dict(step=int(p.stem.split('_')[-1]),success_rate_pct=d['success_rate_pct'],pass_at_4_pct=d['pass_at_4_pct']))
  except (ValueError,KeyError):pass
 s=read(out/'status.json') if (out/'status.json').exists() else {};s.pop('last',None)
 r=dict(experiment=entry['id'],stage=entry.get('stop_step'),pid=child.pid if child else None,
  child_alive=child.poll() is None if child else False,run_status=s,validation=val,
  disk_free_gib={p:shutil.disk_usage(p).free/2**30 for p in ('/home','/data')},unix=time.time())
 atomic(state/'status.json',r);atomic(state/'reports'/f"{entry['id']}.json",r)

def numerical_admission(plan,e,cfg):
 if not cfg.get('vllm_batch_invariant_scoped',False):return
 assert e['kind']=='train','Use the separate isolated numerical-contract runner'
 root=Path(e['requires_scoped_gpu_acceptance'])
 scope=read(root/'scope_acceptance.json');contract=read(root/'acceptance.json')
 assert scope['passed'] and contract['passed']
 assert scope['post_update_inference_then_backward']
 assert scope['native_fp32_and_bf16_autocast_checked']
 helper=Path(plan['scoped_helper'])
 assert sha(helper)==scope['helper_sha256']==plan['source_sha256'][str(helper)]
 assert plan['train_entrypoint'] in plan['source_sha256']
 assert cfg['environment']==read(root/'config.json')['environment']

def run_entry(plan,state,e,fd):
 out=Path(e['output']);out.mkdir(parents=True,exist_ok=True)
 receipt=out/('acceptance.json' if e['kind']=='smoke' else f"stage_{e['stop_step']:04d}.json")
 if receipt.exists() and read(receipt).get('passed'):return True
 if e['kind']=='train' and e['method']=='ucf':
  required=Path(e['requires_gpu_acceptance'])
  if not required.exists() or not read(required).get('passed'):
   atomic(state/'failures'/f"{e['id']}_{e['stop_step']}.json",dict(reason='GPU contract not passed',can_run_baseline=True));return False
 existing=(out/'checkpoint.pt').exists()
 required=8*2**30+((24 if existing else 48)*2**30) if e['kind']=='train' else 8*2**30
 if shutil.disk_usage(out).free<required:
  atomic(state/'failures'/f"{e['id']}_space.json",dict(reason='Insufficient atomic-checkpoint reserve',required_bytes=required,free_bytes=shutil.disk_usage(out).free));return False
 for path,digest in plan['source_sha256'].items():assert sha(path)==digest,'Runtime changed: '+path
 cfg=read(e['config']);assert sha(e['config'])==e['config_sha256']
 numerical_admission(plan,e,cfg)
 assert cfg['model']=='/data/models/Qwen3.5-2B' and cfg['response_protocol']=='first_complete_action_block'
 assert sha(cfg['dataset_path'])==cfg['dataset_sha256']
 if cfg.get('frozen_maps_path'):assert sha(cfg['frozen_maps_path'])==cfg['frozen_maps_sha256']
 runtime=Path(plan['runtime'])
 if e['kind']=='smoke':
  cmd=[plan['python'],'-u',str(runtime/'scripts/v3_gpu_acceptance.py'),'--config',e['config'],'--output',str(out)]
 else:
  entrypoint=plan['train_entrypoint'] if cfg.get('vllm_batch_invariant_scoped',False) else str(runtime/'train.py')
  cmd=[plan['python'],'-u',entrypoint,'--method',e['method'],'--config',e['config'],'--output-dir',str(out),'--stop-after-step',str(e['stop_step'])]
  if existing:
   step=checkpoint_step(out/'checkpoint.pt',plan['python']);assert step<=e['stop_step']
   reconcile(out,step);cmd.append('--resume')
  elif (out/'train_metrics.jsonl').exists():
   reconcile(out,0)
   if (out/'ucf_logs').exists():(out/'ucf_logs').rename(out/f'ucf_logs_failed_{time.time_ns()}')
 name=e['id']+('_smoke' if e['kind']=='smoke' else f"_{e['stop_step']:04d}")
 journal=state/'attempts'/f'{name}.json'
 if journal.exists():
  p=read(journal);assert p.get('start_ticks') is None or identity(p.get('pid',0))!=p['start_ticks'],'Previous child still alive'
 record=dict(command=cmd,started=time.time(),config_sha256=sha(e['config']),kind=e['kind'])
 atomic(out/'launch_config.json',cfg);atomic(journal,record)
 env=dict(os.environ,PYTHONPATH=str(runtime)+':'+str(runtime/'ucf'),UCF_INHERITED_GPU_LOCK_FD=str(fd));env.pop('CUDA_VISIBLE_DEVICES',None)
 with (state/'logs'/f'{name}.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=runtime,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
  record.update(pid=child.pid,start_ticks=identity(child.pid));atomic(journal,record)
  while child.poll() is None:write_progress(state,e,child);time.sleep(20)
  record.update(returncode=child.returncode,ended=time.time());atomic(journal,record)
 write_progress(state,e)
 if child.returncode:
  atomic(state/'failures'/f'{name}.json',dict(**record,log=str(state/'logs'/f'{name}.log'),evidence_retained=True));return False
 try:
  result=read(receipt) if e['kind']=='smoke' else validate_stage(out,e['stop_step'],plan['python'],runtime)
  assert result['passed'];atomic(receipt,result)
  atomic(state/'receipts'/f'{name}.json',dict(passed=True,receipt=str(receipt),sha256=sha(receipt)));return True
 except Exception as exc:
  atomic(state/'failures'/f'{name}.json',dict(verification_error=repr(exc),evidence_retained=True));return False

def main():
 plan=read(sys.argv[1]);state=Path(plan['state'])
 for name in ('logs','receipts','failures','attempts','reports'):(state/name).mkdir(parents=True,exist_ok=True)
 own=(state/'manager.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 keys=('UCF_INHERITED_QUEUE_LOCK_FD','UCF_INHERITED_GPU_LOCK_FD')
 if all(k in os.environ for k in keys):
  locks=[os.fdopen(os.dup(int(os.environ[k])),'a') for k in keys]
 else:
  assert not any(k in os.environ for k in keys),'Incomplete inherited lock pair'
  locks=[Path(p).open('a') for p in plan['locks']]
 for fd,path in zip(locks,plan['locks']):
  assert os.fstat(fd.fileno()).st_ino==Path(path).stat().st_ino
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 failed=set()
 for e in plan['entries']:
  # The supervising agent can arrange a repair at a completed stage boundary.
  # This is never polled inside a live training child and never sends signals.
  if (state/'hold_after_current.json').exists():
   atomic(state/'boundary_held.json',dict(next_entry=e,unix=time.time(),training_interrupted=False))
   return
  if e['id'] in failed:continue
  try:ok=run_entry(plan,state,e,locks[1].fileno())
  except Exception as exc:
   atomic(state/'failures'/f"{e['id']}_exception.json",dict(error=repr(exc),unix=time.time(),evidence_retained=True));ok=False
  if not ok:failed.add(e['id'])
 atomic(state/'queue_finished.json',dict(status='diagnostic_queue_finished',failed=sorted(failed),final_test_used=False,automatic_400_step_promotion=False,unix=time.time()))

if __name__=='__main__':main()
