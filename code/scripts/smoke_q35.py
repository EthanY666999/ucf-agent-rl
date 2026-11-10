"""One-step acceptance only; never saves model weights or starts a formal run."""
import argparse,json,sys,time,hashlib,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
import numpy as np
import train,patr_core as core
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--full-response-budget',action='store_true');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 lock=open('/data/runs/gpu80.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((ROOT/'configs/frozenlake_q35_2b.json').read_text());cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()))
 cfg.update(starpo_keep_fraction=.5,gigpo_gamma=.95,gigpo_omega=1.)
 cfg.update(steps=1,train_batch_size=4,mini_batch_tasks=4,max_response_tokens=1024)
 if a.full_response_budget:cfg['max_response_tokens']=10240
 if a.method in ['tree_random','patr']:cfg['branch_interval']=2
 if a.method in ['grpo','starpo_s','gigpo']:cfg['flat_group_size']=4
 src={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*.py') if 'vendor' not in f.parts and '__pycache__' not in f.parts}
 core.atomic_json(a.output/'hardware.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),model_family='Qwen3.5',model=cfg['model'],environment_freeze='/data/setup_logs/qwen35_freeze_v2_20261107.txt',code_sha256=src))
 core.atomic_json(a.output/'config.json',dict(cfg,method=a.method))
 actor,ref,engine,tok=train.build_models(cfg,cfg['seed']);train.sync_weights(actor,engine)
 tasks=[core.make_datasets()['train'][i] for i in np.random.default_rng(np.random.SeedSequence([42,1,1701])).choice(10000,4,replace=False)]
 rng=np.random.default_rng(42);collector=core.collect
 if a.method=='ucf':
  from ucf_collect import make_collect
  from ucf_weights import training_weights_ucf
  engine._ucf_actor=actor;collector=make_collect(core,core.collect);train.training_weights=training_weights_ucf
  cfg.update(ucf_beta=.25,ucf_variant='ucf',ucf_c0=1.,ucf_controller='v2',ucf_log_dir=str(a.output/'ucf_logs'),_step=1)
 if a.method in ['starpo_s','gigpo']:
  from external_baselines import configure
  configure(train,cfg,a.method)
 started=time.monotonic();bs,stats,events=collector(tasks,engine,tok,cfg,'grpo' if a.method in ['starpo_s','gigpo'] else a.method,rng)
 sampled=time.monotonic()-started
 counts={str(t):sum(x==t for b in bs for x,m in zip(b.ids,b.mask) if m) for t in [248068,248069]}
 train.write_rollouts(a.output/'rollouts.jsonl.gz',bs,tasks)
 core.atomic_json(a.output/'sampled.json',dict(generated_think_tokens=counts,stats=stats,events=events))
 assert not any(counts.values()),counts
 optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 changes=train.update(actor,ref,optimizer,bs,cfg,a.method,tok,np.random.default_rng(np.random.SeedSequence([42,1,1702])),len(tasks))
 train.sync_weights(actor,engine)
 train.write_rollouts(a.output/'rollouts.jsonl.gz',bs,tasks)
 core.atomic_json(a.output/'events.json',events)
 if a.method=='ucf':
  budget=stats['fork_budget']
  assert abs(budget['expected_forks']-min(2.,budget['reachable_max']))<1e-8,budget
  if not budget['capacity_limited']:assert 1<=stats['mean_forks']<=3,stats['mean_forks']
  else:stats['original_1_to_3_smoke_gate_capacity_exception']=True
 result=dict(passed=True,method=a.method,generated_think_tokens=counts,weight_sync_assertion=True,rollout_seconds=sampled,tokens_per_second=stats['generated_tokens']/sampled,step_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,stats=stats,changes=changes,smoke_weights_discarded=True)
 if a.method=='ucf':
  from ucf_gpu_fixture import verify
  fixture_start=time.monotonic()
  result['forced_decoder_fixture']=verify(core,train,actor,engine,tok,cfg,core.make_datasets()['train'][:100])
  result['forced_decoder_fixture_seconds']=time.monotonic()-fixture_start
 core.atomic_json(a.output/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
