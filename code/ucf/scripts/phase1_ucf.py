"""Production CPU-compatible gradient diagnostic helpers; deployment harness omitted."""
import argparse,copy,json,math,os,sys,time
from pathlib import Path
import numpy as np
import patr_core as core
from cert_collect import continuations
from cert_select import stable_seed,deduplicate,select
from ucf_collect import initial_ucf,counterfactual
from ucf_select import select_all
from ucf_weights import pseudo
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
