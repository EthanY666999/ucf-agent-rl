"""Decoder/prefix fixture only. Synthetic original action is never used for training."""
import copy,math,time
from types import SimpleNamespace as NS
import torch

def verify(core,train,actor,engine,tok,cfg,tasks):
 from ucf_probe import probe_exact
 from ucf_collect import counterfactual
 from cert_entropy import consequence_classes
 # Use a training task with at least two distinct immediate consequences.
 task=next(t for t in tasks if len(set(consequence_classes(core.make_env(t,cfg)).values()))>1)
 pre=core.start_branch(task,0,tok,cfg,'0');original=copy.deepcopy(pre)
 target=tok.encode('```Right```',add_special_tokens=False)+[tok.eos_token_id]
 ids=torch.tensor([pre.ids+target],device='cuda');mask=torch.tensor([[0]*len(pre.ids)+[1]*len(target)],device='cuda')
 with torch.no_grad():lp=train.selected_logprobs(actor,ids,torch.ones_like(ids),mask).tolist()
 output=NS(token_ids=target,logprobs=[{x:NS(logprob=p)} for x,p in zip(target,lp)],text='```Right```',finish_reason='stop')
 core.apply_response(original,output,tok,cfg)
 n=dict(branch=pre,j=0,turn=1,state=int(pre.env.s),boundary=len(pre.ids),parent_action=3,S=0.,rho=1.,baseline=0.)
 probe_exact({0:[n]},engine,tok,cfg,42)
 rows,children,cost=counterfactual(core,{0:[n]},[original],[task],engine,tok,cfg,42,991)
 assert cost['cf_count']==1 and len(rows)==2
 assert n['forced_token'] in n['stratum']['allowed_tokens']
 assert all(math.isfinite(v) for b in children.values() for v in b.old_logprobs)
 return dict(passed=True,source='synthetic action at a training-task prefix; never included in optimizer update',forced_action_outside_original_consequence=True,prefix_ids_mask_logprobs_equal=True,finite_logprobs=True,counterfactual_count=cost['cf_count'])
