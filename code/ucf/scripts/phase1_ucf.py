"""Fixed training tasks, independent references and paired gradient diagnostics."""
import argparse,copy,json,math,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from cert_collect import continuations
from cert_select import stable_seed,deduplicate,select
from ucf_collect import initial_ucf,counterfactual
from ucf_select import select_all
from ucf_weights import pseudo
MODELS={'sft':'/home/vipuser/golden-sft-patr-20261031/code/runs/golden_sft/sft/final_model','grpo400':'/home/vipuser/golden-sft-patr-20261031/code/runs/formal/grpo_seed42/final_model','random400':'/home/vipuser/golden-next-20261031/b3/code/runs/formal/tree_random_seed42/final_model'}
def loo_rows(branches,n):
 rows=[]
 for ti in range(n):
  bs=[b for b in branches if b.task_id==ti];k=len(bs);assert k>1
  total=sum(b.reward for b in bs)
  for b in bs:rows.append((b,b.reward-(total-b.reward)/(k-1),1/(n*k)))
 return rows

def grad(actor,rows,cfg,tok):
 import torch,train
 was_training=actor.training
 assert getattr(actor.config,'attention_dropout',0.)==0.
 actor.train()  # Required for the frozen model's activation checkpointing.
 actor.zero_grad(set_to_none=True)
 for chunk in train.microbatches(rows,cfg):
  ids,attn,mask,adv,w,_=train.pack(chunk,tok.pad_token_id)
  lp=train.selected_logprobs(actor,ids,attn,mask,gradient=True);loss=(w*adv*lp).sum();assert torch.isfinite(loss);loss.backward()
  del ids,attn,mask,adv,w,lp,loss
 g=torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.numel(),device=p.device) for p in actor.parameters()]).float()
 assert torch.isfinite(g).all();actor.zero_grad(set_to_none=True);actor.train(was_training)
 return g

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);a=p.parse_args()
 os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']));assert json.loads((ROOT/'artifacts/gpu_acceptance.json').read_text())['passed']
 import torch,train
 from ucf_runtime import configure_builder
 configure_builder(train)
 cfg=json.loads((ROOT/'configs/base.json').read_text());cfg.update(model=MODELS[a.model],ucf_beta=1.,ucf_variant='ucf')
 c0=json.loads((ROOT/'artifacts/calibration.json').read_text())['c0'];out=ROOT/'runs/phase1'/a.model;out.mkdir(parents=True,exist_ok=True)
 data=json.loads((ROOT/'artifacts/data.json').read_text())['train'];actor,ref,engine,tok=train.build_models(cfg,42);actor.eval();del ref
 os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
 for bi in range(3):
  tasks=[data[i] for i in np.random.default_rng(np.random.SeedSequence([42,bi+1,1701])).choice(len(data),64,replace=False)]
  for rep in range(4):
   dest=out/f'b{bi+1}_r{rep+1}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():continue
   for stale in dest.glob('selection.jsonl'):
    stale.rename(stale.with_name(f'selection_incomplete_{time.time_ns()}.jsonl'))
   seed=stable_seed(42,'phase1',bi,rep);core.atomic_json(out/'status.json',dict(stage='sampling',batch=bi+1,repeat=rep+1,updated=time.time()))
   first,candidates,icost,pcost=initial_ucf(core,tasks,engine,tok,cfg,seed)
   picked,selection=select_all(candidates,first,tasks,c0,seed,bi*4+rep,dest/'selection.jsonl')
   cfrows,cfchildren,cfcost=counterfactual(core,picked,first,tasks,engine,tok,cfg,seed,bi*4+rep)
   assert cfcost['cf_parse_mismatch']<=.02,'CF parser diagnostic exceeds acceptance limit'
   plans={};union={ti:{(n['j'],n['turn']):n for n in ns} for ti,ns in picked.items()}
   for mode in ['uct','uniform']:
    plans[mode]={}
    for ti,ns in candidates.items():
     ph=sum(b.reward for b in first if b.task_id==ti)/4
     plans[mode][ti]=select(deduplicate(ns),mode,2,ph,seed,tasks[ti]['uid'],bi*4+rep)
     for n in plans[mode][ti]:union[ti][(n['j'],n['turn'])]=n
   normal,ncost,skips=continuations(core,{ti:list(v.values()) for ti,v in union.items()},tasks,engine,tok,cfg,seed);assert not skips
   refs,refcost,_=core.collect(tasks,engine,tok,dict(cfg,initial_branches=32,branch_interval=cfg['max_turns']+1),'patr',np.random.default_rng(stable_seed(seed,'independent-reference')))
   assert len(refs)==64*32
   for label,bs in [('initial',first),('reference',refs),('normal',list(x[0] for x in normal.values())),('cf',list(cfchildren.values()))]:train.write_rollouts(dest/(label+'.jsonl.gz'),bs,tasks)
   core.atomic_json(out/'status.json',dict(stage='gradients',batch=bi+1,repeat=rep+1,updated=time.time()))
   refg=grad(actor,loo_rows(refs,64),cfg,tok)
   ref1=grad(actor,loo_rows([b for b in refs if int(b.lineage)<16],64),cfg,tok)
   ref2=grad(actor,loo_rows([b for b in refs if int(b.lineage)>=16],64),cfg,tok)
   targets=[refg,ref1,ref2];norms=[float((g*g).sum()) for g in targets]
   def metrics(g):
    norm=float((g*g).sum());dots=[float((g*x).sum()) for x in targets]
    return dict(norm2=norm,dots=dots,cosines=[d/math.sqrt(norm*v) if norm>0 and v>0 else None for d,v in zip(dots,norms)],relative_mse=[max(0.,(norm-2*d+v)/v) if v>0 else None for d,v in zip(dots,norms)])
   A=grad(actor,loo_rows(first,64),cfg,tok);results={'GRPO4':metrics(A)}
   raw=[];ht=[]
   for ti,ns in picked.items():
    for n in ns:
     b=normal[(ti,n['j'],n['turn'])][0];v=b.reward-n['baseline'];pb=pseudo(b,n['boundary'],'phase1',v,n['rho'],1,1,'ucf')
     raw.append((pb,v,1/(64*4)));ht.append((pb,v,1/(64*4*n['rho'])))
   cf=[(b,b.ucf['adv']/2,1/(64*4*b.ucf['rho'])) for b in cfrows]
   for name,rows in [('Braw',raw),('Bht',ht),('Bcf',cf)]:
    B=grad(actor,rows,cfg,tok);results[name]=metrics(B)
    for beta in [.25,.5,1.]:results[f'A+{beta:g}*{name}']=metrics((A+beta*B)/(1+beta))
    del B
   del A
   for mode,plan in plans.items():
    leaves=first+[normal[(ti,n['j'],n['turn'])][0] for ti,ns in plan.items() for n in ns]
    G=grad(actor,loo_rows(leaves,64),cfg,tok);results['CERT-'+mode]=metrics(G);del G
   noise=float((ref1*ref2).sum())/math.sqrt(norms[1]*norms[2]) if norms[1]>0 and norms[2]>0 else None
   core.atomic_json(dest/'result.json',dict(model=a.model,batch=bi+1,repeat=rep+1,seed=seed,estimators=results,reference_norm2=norms,half_reference_cosine=noise,reference_interpretation='Finite independent 32-sample Monte Carlo reference; half agreement is reliability diagnostic, not literal noise ceiling',cost=dict(initial=icost,probe=pcost,cf=cfcost,normal_shared_union=ncost,reference=refcost)))
   del refg,ref1,ref2,targets,refs,first,cfrows,cfchildren,normal;torch.cuda.empty_cache()
 core.atomic_json(out/'status.json',dict(status='complete',updated=time.time()))
if __name__=='__main__':main()
