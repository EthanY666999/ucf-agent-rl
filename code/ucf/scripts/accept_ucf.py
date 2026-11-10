"""Train-only calibration and technical acceptance; smoke changes never exported."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from cert_select import stable_seed
from ucf_collect import initial_ucf,counterfactual
from ucf_select import calibrate,feature,select_all
from ucf_weights import training_weights_ucf

def describe(xs):
 a=np.array(xs,dtype=float);return dict(n=len(a),mean=float(a.mean()) if len(a) else None,quantiles=np.quantile(a,[0,.25,.5,.75,.95,1]).tolist() if len(a) else [])
def component_grad(actor,ref,branches,cfg,tok,part):
 import torch,train
 was_training=actor.training
 assert getattr(actor.config,'attention_dropout',0.)==0.,'Diagnostic train-mode requires deterministic dropout-free actor'
 actor.train()  # Enables the existing gradient-checkpointing path.
 rows=training_weights_ucf(branches,len({b.task_id for b in branches}));rows=[r for r in rows if hasattr(r[0],'ucf')==(part=='B')]
 actor.zero_grad(set_to_none=True);loss_total=0.
 for chunk in train.microbatches(rows,cfg):
  ids,attn,mask,adv,w,old=train.pack(chunk,tok.pad_token_id)
  current=train.selected_logprobs(actor,ids,attn,mask,gradient=True)
  with torch.no_grad():ref_lp=train.selected_logprobs(ref,ids,attn,mask)
  ratio=(current-old).exp();pg=-torch.minimum(ratio*adv,ratio.clamp(1-cfg['clip_low'],1+cfg['clip_high'])*adv)
  lr=ref_lp-current;kl=lr.exp()-lr-1;loss=((pg+cfg['kl_coefficient']*kl)*w).sum();loss.backward();loss_total+=loss.item()
  del ids,attn,mask,adv,w,old,current,ref_lp,ratio,pg,lr,kl,loss
 norm=math.sqrt(sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None))
 assert math.isfinite(norm) and math.isfinite(loss_total);actor.zero_grad(set_to_none=True)
 actor.train(was_training)
 return dict(loss=loss_total,grad_norm=norm)
def main():
 os.fstat(int(os.environ['NEXT_GPU_LEASE_FD']));assert json.loads((ROOT/'artifacts/cpu_acceptance.json').read_text())['passed']
 import torch,train
 from ucf_runtime import configure_builder
 configure_builder(train)
 cfg=json.loads((ROOT/'configs/base.json').read_text());cfg.update(ucf_beta=1.,ucf_variant='ucf')
 out=ROOT/'runs/acceptance';out.mkdir(parents=True,exist_ok=True)
 data=json.loads((ROOT/'artifacts/data.json').read_text())['train'];batches=[[data[i] for i in np.random.default_rng(np.random.SeedSequence([42,b+1,1701])).choice(len(data),64,replace=False)] for b in range(3)]
 actor,ref,engine,tok=train.build_models(cfg,42);weights=[];summaries=[]
 os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
 for bi,tasks in enumerate(batches):
  core.atomic_json(out/'status.json',dict(stage='calibration',batch=bi+1,updated=time.time()))
  base=stable_seed(42,'ucf-calibration',bi)
  first,candidates,icost,pcost=initial_ucf(core,tasks,engine,tok,cfg,base)
  ws=[[feature(n,first)[0] for n in candidates[i]] for i in range(64)];weights.extend(ws)
  summaries.append(dict(batch=bi+1,weights=ws,initial_cost=icost,probe_cost=pcost))
  core.atomic_json(out/f'calibration_{bi+1}.json',summaries[-1])
 c0=calibrate(weights);core.atomic_json(ROOT/'artifacts/calibration.json',dict(c0=c0,c0_uniform=calibrate([[1.]*len(ws) for ws in weights]),mean_expected_forks=2.,batches=summaries,train_only=True))
 tasks=batches[0];seed=stable_seed(42,'ucf-acceptance')
 first,candidates,icost,pcost=initial_ucf(core,tasks,engine,tok,cfg,seed)
 picked,selection=select_all(candidates,first,tasks,c0,seed,0,out/'selection.jsonl')
 pseudo,children,ccost=counterfactual(core,picked,first,tasks,engine,tok,cfg,seed,0)
 ns=[n for nodes in picked.values() for n in nodes];cfns=[n for n in ns if not n['stratum']['cf_fallback']]
 mean_forks=len(ns)/64
 checks=dict(finite_logprobs=all(math.isfinite(v) for b in first+pseudo+list(children.values()) for v in b.old_logprobs),cf_parse_mismatch=ccost['cf_parse_mismatch']<=.02,mean_forks=1<=mean_forks<=3)
 report=dict(probability_source='actor full vocabulary; no vLLM prompt scoring',actor_vllm_observed_difference=describe(pcost['actor_vllm_logprob_differences']),actor_vs_vllm_forced_conditional_difference=describe([n['actor_vs_vllm_forced_conditional_difference'] for n in cfns]),checks=checks,c0=c0,actual_mean_forks=mean_forks,cf_fallback=sum(n['stratum']['cf_fallback'] for n in ns)/max(1,len(ns)),unsampled_mass=describe([n['stratum']['unsampled_mass'] for n in ns]),pi_o=describe([n['stratum']['pi_o'] for n in ns]),score_probe_difference=describe([v for n in cfns for v in n['score_probe_difference']]),initial_cost=icost,probe_cost=pcost,cf_cost=ccost)
 core.atomic_json(out/'sampling_report.json',report)
 assert all(checks.values()),checks
 branches=first+pseudo;train.training_weights=training_weights_ucf
 report['components']={p:component_grad(actor,ref,branches,cfg,tok,p) for p in ['A','B']}
 optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 metrics=train.update(actor,ref,optimizer,branches,cfg,'patr',tok,np.random.default_rng(np.random.SeedSequence([42,1,1702])),64)
 assert math.isfinite(metrics['grad_norm']) and metrics['max_parameter_change']>0
 report.update(smoke=metrics,passed=True,smoke_weights_discarded=True)
 core.atomic_json(ROOT/'artifacts/gpu_acceptance.json',report);core.atomic_json(out/'status.json',dict(status='passed',updated=time.time()))
if __name__=='__main__':main()
