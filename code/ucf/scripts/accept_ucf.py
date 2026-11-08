"""Production CPU-compatible gradient diagnostic helpers; deployment harness omitted."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
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
