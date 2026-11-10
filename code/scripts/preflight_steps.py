"""Stage-0 regression/cost only, isolated checkpoints, no formal queue admission."""
import argparse,sys,json,time,hashlib,subprocess,fcntl,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['crosscard','cost'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
ROOT=Path('/data/decision-point/code');out=Path('/data/runs/stage0')/a.kind
source=out/'source' if a.kind=='crosscard' else ROOT
sys.path.insert(0,str(source))
import numpy as np
import torch
import train,patr_core as core
out.mkdir(parents=True,exist_ok=True)
lock=open('/data/runs/gpu80.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if a.kind=='crosscard':
 cfg=json.loads((source/'runs/formal/grpo_seed42/config.json').read_text());cfg['model']='/data/models/Qwen2.5-0.5B-Instruct';nsteps=10
else:
 cfg=json.loads((ROOT/'configs/frozenlake_q35_2b.json').read_text());cfg.update(json.loads((ROOT/'artifacts/runtime_override.json').read_text()));cfg['flat_group_size']=4;nsteps=5
assert cfg['train_batch_size']==64 and cfg['max_response_tokens']==10240
core.atomic_json(out/'config.json',cfg)
core.atomic_json(out/'hardware.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),model_family='Qwen2.5' if a.kind=='crosscard' else 'Qwen3.5',source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['train.py','patr_core.py']},environment_freeze='/data/setup_logs/patr_freeze.txt' if a.kind=='crosscard' else '/data/setup_logs/qwen35_freeze_v2_20261107.txt',purpose=a.kind))
os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
actor,ref,engine,tok=train.build_models(cfg,42)
if a.kind=='crosscard':
 from legacy_vocab_validation import install
 install(engine,actor.config.vocab_size,out/'legacy_vocab_events.jsonl')
optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
rng=np.random.default_rng(42);begin=0;cp=out/'checkpoint.pt'
if a.resume:
 state=torch.load(cp,map_location='cpu',weights_only=False)
 actor.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];begin=state['step'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
 train.sync_weights(actor,engine)
data=core.make_datasets();started=time.monotonic()
for step in range(begin+1,nsteps+1):
 core.atomic_json(out/'status.json',dict(status='running',kind=a.kind,current_step=step,total_steps=nsteps,pid=os.getpid()))
 tasks=[data['train'][i] for i in np.random.default_rng(np.random.SeedSequence([42,step,1701])).choice(len(data['train']),64,replace=False)]
 start=time.monotonic();bs,stats,events=core.collect(tasks,engine,tok,cfg,'grpo',rng);rollout=time.monotonic()-start
 lengths=[b.ntokens for b in bs]
 think={str(t):sum(x==t for b in bs for x,k in zip(b.ids,b.mask) if k) for t in [248068,248069]} if a.kind=='cost' else {}
 # The zero-think assertion is a smoke-test gate, not part of the cost protocol.
 # Preserve long-budget observations instead of silently changing decoding support.
 train.write_rollouts(out/f'rollouts_{step:04d}.jsonl.gz',bs,tasks)
 core.atomic_json(out/f'sampled_{step:04d}.json',dict(step=step,rollout_seconds=rollout,trajectory_response_lengths=lengths,generated_think_tokens=think,stats=stats))
 changes=train.update(actor,ref,optimizer,bs,cfg,'grpo',tok,np.random.default_rng(np.random.SeedSequence([42,step,1702])),64);train.sync_weights(actor,engine)
 record=dict(step=step,step_seconds=time.monotonic()-start,rollout_seconds=rollout,trajectory_response_lengths=lengths,length_median=float(np.median(lengths)),length_p95=float(np.quantile(lengths,.95)),hit_response_limit_fraction=sum(n>=10240 for n in lengths)/len(lengths),generated_think_tokens=think,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**stats,**changes)
 core.append_json(out/'train_metrics.jsonl',record);print(json.dumps(record),flush=True)
 train.save_checkpoint(cp,actor,optimizer,step,rng,cfg,time.monotonic()-started)
 core.atomic_json(out/'status.json',dict(status='running',kind=a.kind,step=step,total_steps=nsteps,pid=os.getpid(),last=record))
records=[json.loads(l) for l in (out/'train_metrics.jsonl').read_text().splitlines()]
by_step={r['step']:r for r in records};assert set(by_step)==set(range(1,nsteps+1));records=list(by_step.values())
if a.kind=='cost':
 lengths=[n for r in records for n in r['trajectory_response_lengths']];p95=float(np.quantile(lengths,.95))
 result=dict(passed=True,steps=5,batch_tasks=64,trajectories_per_task=4,measured_max_response_tokens=10240,trajectory_length_median=float(np.median(lengths)),trajectory_length_p95=p95,hit_response_limit_fraction=sum(n>=10240 for n in lengths)/len(lengths),nosft_max_response_tokens=2048 if p95<=2048 else 4096,sft_max_response_tokens=10240,step_seconds=[r['step_seconds'] for r in records],selection_rule='user preregistered p95<=2048 => 2048; otherwise 4096',weights_not_reused_for_formal=True)
else:
 reference=[json.loads(l) for l in (source/'runs/formal/grpo_seed42/train_metrics.jsonl').read_text().splitlines()];refrows={r['step']:r for r in reference};pairs=[]
 for r in records:
  q=refrows[r['step']];pairs.append(dict(step=r['step'],**{k:dict(current=r[k],reference=q[k],ratio=r[k]/q[k] if q[k] else None) for k in ['success_rate','kl','grad_norm']}))
 result=dict(status='awaiting_comparability_review',steps=10,pairs=pairs,reference_source='40G host historical patr-reproduction GRPO records; original-run GPU provenance must be checked before claiming same-card identity')
core.atomic_json(out/'result.json',result);core.atomic_json(out/'status.json',dict(status='complete',kind=a.kind,steps=nsteps,result=result));print(json.dumps(result),flush=True)
