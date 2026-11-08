"""UCF adapter; original files remain byte-identical to B2/B3/G2."""
import copy,gzip,json,math,os,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from cert_collect import initial,replay,stream
from cert_select import stable_seed
from tokens import fence_id
from cert_entropy import token_map
from ucf_probe import probe_exact,requests,forced_logprobs
from ucf_select import Controller,select_all,stratum,feature,calibrate
from ucf_weights import fork_rows,generation_end

def initial_ucf(core,tasks,engine,tokenizer,cfg,seed):
 first,nodes,cost,_=initial(core,tasks,engine,tokenizer,cfg,seed,topk=1,with_probes=False)
 candidates={ti:[n for n in nodes if n['branch'].task_id==ti and n['parent_action'] is not None] for ti in range(len(tasks))}
 by_branch={(b.task_id,int(b.lineage)):b for b in first}
 mapping,_=token_map(tokenizer);fence=fence_id(tokenizer)
 for ti,ns in candidates.items():
  for n in ns:
   b=by_branch[(ti,n['j'])];bd=n['boundary'];obs={}
   if b.ids[bd]==fence:
    obs['fence']=b.old_logprobs[bd]
    if bd+1<len(b.ids) and b.mask[bd+1] and b.ids[bd+1] in mapping:obs.update(action_token=b.ids[bd+1],action=b.old_logprobs[bd+1])
   n['vllm_observed']=obs
 probe=probe_exact(candidates,engine,tokenizer,cfg,seed)
 return first,candidates,cost,probe

def counterfactual(core,picked,first,tasks,engine,tokenizer,cfg,seed,step):
 from vllm import SamplingParams
 mapping,_=token_map(tokenizer);fence=fence_id(tokenizer);orig={(b.task_id,int(b.lineage)):b for b in first};children={};jobs=[];node_map={};rows=[]
 for ti,ns in picked.items():
  for n in ns:
   o=orig[(ti,n['j'])];bd=n['boundary'];end=generation_end(o,bd)
   s=stratum(n,o.ids[bd:end],mapping,fence);n['stratum']=s
   s['parse_mismatch']=o.ids[bd]!=fence or s['first_action']!=n['parent_action']
   assert not s['parse_mismatch'],'Excluded original format mismatch reached branching'
   key=(ti,n['j'],n['turn']);node_map[key]=n
   if s['cf_fallback']:continue
   b=replay(core,n,tasks[ti],cfg);b.lineage=f"{n['j']}@{n['turn']}.cf";children[key]=b
   rem=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
   assert rem>=3,'Insufficient budget for fence, first action token and remainder'
   jobs.append((key,b.ids+[fence],SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,allowed_token_ids=s['allowed_tokens'],logprobs=1,seed=stable_seed(seed,tasks[ti]['uid'],n['j'],n['turn'],step,'cf'))))
 forced,fcost=requests(engine,jobs);tail_jobs=[]
 for key,b in children.items():
  n=node_map[key];x=int(forced[key].outputs[0].token_ids[0]);n['forced_token']=x
  assert x in n['stratum']['allowed_tokens'] and mapping[x] in n['stratum']['allowed_actions']
  ids=b.ids+[fence,x]
  rem=min(cfg['max_response_tokens']-b.ntokens-2,cfg['max_context_tokens']-len(ids))
  tail_jobs.append((key,ids,SamplingParams(temperature=cfg['temperature'],top_p=1.,top_k=-1,max_tokens=rem,logprobs=1,seed=stable_seed(seed,tasks[b.task_id]['uid'],b.lineage,b.turns),stop_token_ids=[tokenizer.eos_token_id])))
 scost=dict(input_tokens=0,generated_tokens=0,request_seconds=0.,seconds=0.,source='reuse actor prefix+fence probabilities computed before selection')
 tails,tcost=requests(engine,tail_jobs);active=[];done={};mismatch=0
 for key,b in children.items():
  n=node_map[key];x=n['forced_token'];lp=forced_logprobs(n,fence,x);probe=n['exact_logprobs'][x]
  n['score_probe_difference']=[lp[i]-probe[i] for i in range(2)]
  restricted=float(forced[key].outputs[0].logprobs[0][x].logprob)
  allowed_mass=sum(math.exp(n['exact_logprobs'][t][1]) for t in n['stratum']['allowed_tokens'])
  n['actor_vs_vllm_forced_conditional_difference']=lp[1]-math.log(allowed_mass)-restricted
  n['forced_vllm_logprob_restricted']=restricted
  tail=tails[key].outputs[0];ids=[fence,x]+list(tail.token_ids)
  out=SimpleNamespace(token_ids=ids,logprobs=[{fence:SimpleNamespace(logprob=lp[0])},{x:SimpleNamespace(logprob=lp[1])}]+list(tail.logprobs),text=tokenizer.decode(ids,skip_special_tokens=True),finish_reason=tail.finish_reason)
  core.apply_response(b,out,tokenizer,cfg)
  parsed=b.actions[-1] if b.turns>=n['turn'] else None
  n['cf_parse_mismatch']=parsed!=mapping[x];mismatch+=n['cf_parse_mismatch']
  bd=n['boundary']
  for attr in ['ids','mask','old_logprobs']:assert getattr(b,attr)[:bd]==getattr(n['branch'],attr)
  assert all(math.isfinite(v) for v in b.old_logprobs)
  if not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:b.terminal='pruned_repeat';b.reward=0.
  if b.terminal:done[key]=b
  else:active.append(b)
 leaves,_,continuation_cost,_=stream(core,active,tasks,engine,tokenizer,cfg,seed,topk=1,capture=False)
 reverse={(b.task_id,b.lineage):k for k,b in children.items()}
 for b in leaves:done[reverse[(b.task_id,b.lineage)]]=b
 for key,n in node_map.items():
  o=orig[(key[0],key[1])];cf=done.get(key)
  rows.extend(fork_rows(o,cf,n,cfg['ucf_beta'],cfg['ucf_variant']))
  n['R_o']=o.reward;n['R_cf']=None if cf is None else cf.reward
 cost=dict(forced=fcost,scoring=scost,tail=tcost,continuation=continuation_cost,cf_parse_mismatch=mismatch/max(1,len(children)),cf_count=len(children))
 return rows,done,cost

def make_collect(core,original):
 def collect(tasks,engine,tokenizer,cfg,method,rng,evaluation=False):
  if evaluation:return original(tasks,engine,tokenizer,cfg,method,rng,evaluation=True)
  assert cfg['temperature']==1. and cfg['initial_branches']==4
  out=Path(cfg['ucf_log_dir']);out.mkdir(parents=True,exist_ok=True)
  step=cfg.get('_step',0)
  progress=os.environ.get('PATR_PROGRESS_PATH')
  if progress and (Path(progress).parent/'status.json').exists():step=json.loads((Path(progress).parent/'status.json').read_text()).get('current_step',step)
  state=out/'controller.json'
  data=json.loads(state.read_text()) if state.exists() else dict(c=cfg['ucf_c0'],history=[],last_step=-1)
  assert step>data['last_step'],'Refuse ambiguous resume: reconcile controller with checkpoint before retry'
  ctrl=Controller(data['c'],data['history']);c_step=ctrl.c
  seed=int(rng.integers(0,2**31-1))
  first,candidates,icost,pcost=initial_ucf(core,tasks,engine,tokenizer,cfg,seed)
  # Part B estimates only nodes whose original response begins with a parsed fence.
  mapping,_=token_map(tokenizer);fence=fence_id(tokenizer)
  originals={(b.task_id,int(b.lineage)):b for b in first}
  for ti,ns in candidates.items():
   for n in ns:
    original_branch=originals[(ti,n['j'])];bd=n['boundary'];end=generation_end(original_branch,bd)
    st=stratum(n,original_branch.ids[bd:end],mapping,fence)
    st['parse_mismatch']=original_branch.ids[bd]!=fence or st['first_action']!=n['parent_action']
    n['stratum']=st
  if cfg.get('ucf_controller','v2')=='v2':
   task_weights=[[feature(n,first,cfg['ucf_variant'])[0] for n in ns if not n['stratum']['parse_mismatch']] for ns in candidates.values()]
   c_step=calibrate(task_weights,target=2.)
   budget=dict(target=2.,reachable_max=sum(sum(1. if w>0 else .03 for w in ws) for ws in task_weights)/len(tasks),expected_forks=sum(sum(min(1.,max(.03,c_step*w)) for w in ws) for ws in task_weights)/len(tasks))
   budget['capacity_limited']=budget['reachable_max']<=2.
   core.atomic_json(out/f'budget_{step:04d}.json',budget)
  picked,selection=select_all(candidates,first,tasks,c_step,seed,step,out/f'select_{step:04d}.jsonl',cfg['ucf_variant'])
  pseudo,children,ccost=counterfactual(core,picked,first,tasks,engine,tokenizer,cfg,seed,step)
  # Preserve entire initial and CF leaves, including untrained continuation tokens.
  with gzip.open(out/f'raw_leaves_{step:04d}.jsonl.gz','xt') as evidence:
   for kind,bs in [('initial',first),('counterfactual',list(children.values()))]:
    for b in bs:evidence.write(json.dumps(dict(kind=kind,task=tasks[b.task_id],task_id=b.task_id,lineage=b.lineage,ids=b.ids,mask=b.mask,old_logprobs=b.old_logprobs,actions=b.actions,trace=b.trace,reward=b.reward,terminal=b.terminal))+'\n')
  nsel=sum(map(len,picked.values()))
  if cfg.get('ucf_controller','v2')=='v1':ctrl.finish(nsel/len(tasks))
  else:ctrl.c=c_step;ctrl.history.append(nsel/len(tasks))
  controller_state=dict(c=ctrl.c,c_step=c_step,mean_forks=nsel/len(tasks),history=ctrl.history,last_step=step)
  core.atomic_json(out/f'controller_{step:04d}.json',controller_state)
  core.atomic_json(state,controller_state)
  events=[]
  for ti,ns in picked.items():
   for n in ns:events.append(dict(kind='ucf_fork',task_id=ti,**{k:v for k,v in n.items() if k not in ['branch','exact_logprobs']}))
  leaves=first+list(children.values());counts=Counter(b.terminal for b in leaves)
  mixed=sum(len({b.reward for b in first if b.task_id==i})>1 for i in range(len(tasks)))
  stats=dict(episodes=len(leaves),success_rate=sum(b.reward>0 for b in first)/len(first),mean_response_tokens=sum(b.ntokens for b in first)/len(first),mean_turns=sum(b.turns for b in first)/len(first),terminal_counts=dict(counts),mixed_groups=mixed,task_groups=len(tasks),mixed_group_fraction=mixed/len(tasks),branch_events=nsel,score_prunes=0,c_step=c_step,c_next=ctrl.c,mean_forks=nsel/len(tasks),fork_counts=[len(picked[i]) for i in range(len(tasks))],probe_cost=pcost,cf_cost=ccost,rollout_sampled_surprisal=-sum(lp for b in first for lp,m in zip(b.old_logprobs,b.mask) if m)/sum(b.ntokens for b in first))
  for k in ['generated_tokens','input_tokens']:stats[k]=icost[k]+pcost[k]+sum(ccost[a][k] for a in ['forced','scoring','tail','continuation'])
  # Forced fence token is generated through conditioning and reported separately.
  if cfg.get('ucf_controller','v2')=='v2':stats['fork_budget']=budget
  stats['conditioned_fence_tokens']=ccost['cf_count']
  stats['executed_actions']=icost['executed_actions']+sum(max(0,b.turns-n['branch'].turns) for ti,ns in picked.items() for n in ns for b in [children.get((ti,n['j'],n['turn']))] if b is not None)
  stats['invalid_actions']=icost['invalid_actions']+sum(sum(a==0 for a in b.actions[n['branch'].turns:]) for ti,ns in picked.items() for n in ns for b in [children.get((ti,n['j'],n['turn']))] if b is not None)
  stats['invalid_action_fraction']=stats['invalid_actions']/max(1,stats['executed_actions'])
  ns=[n for nodes in picked.values() for n in nodes];cfns=[n for n in ns if not n['stratum']['cf_fallback']]
  stats.update(rho_values=[r['rho'] for r in selection],selected_turn_histogram=dict(Counter(n['turn'] for n in ns)),pi_o_values=[n['stratum']['pi_o'] for n in ns],unsampled_mass_values=[n['stratum']['unsampled_mass'] for n in ns],cf_fallback_fraction=sum(n['stratum']['cf_fallback'] for n in ns)/max(1,len(ns)),parse_mismatch_fraction=sum(n['stratum']['parse_mismatch'] for n in ns)/max(1,len(ns)),cf_parse_mismatch_fraction=ccost['cf_parse_mismatch'],counterfactual_flip_rate=sum(n['R_o']!=n['R_cf'] for n in cfns)/len(cfns) if cfns else None)
  import numpy as np
  groups={i:sum(b.reward>0 for b in first if b.task_id==i) for i in range(len(tasks))}
  types={i:('all_failure' if n==0 else 'all_success' if n==4 else 'mixed') for i,n in groups.items()}
  group_counts=dict(Counter(types.values()))
  failures=[i for i,k in types.items() if k=='all_failure']
  rescued=sum(any(b.task_id==i and b.reward>0 for b in children.values()) for i in failures)
  all_nodes=[n for ns in candidates.values() for n in ns]
  stats.update(initial_group_counts={k:group_counts.get(k,0) for k in ['all_failure','all_success','mixed']},
   initial_group_fractions={k:group_counts.get(k,0)/len(tasks) for k in ['all_failure','all_success','mixed']},
   rescue_rate=rescued/len(failures) if failures else None,
   original_parse_mismatch_fraction=sum(n['stratum']['parse_mismatch'] for n in all_nodes)/max(1,len(all_nodes)),
   counterfactual_parse_mismatch_fraction=ccost['cf_parse_mismatch'],
   forks_p10_p50_p90=np.quantile(stats['fork_counts'],[.1,.5,.9]).tolist(),
   flip_rate_by_group={k:sum(n['R_o']!=n['R_cf'] for n in cfns if types[n['branch'].task_id]==k)/max(1,sum(types[n['branch'].task_id]==k for n in cfns)) for k in ['all_failure','all_success','mixed']})
  from scripts.hard_metrics import leaf_group_counts
  stats.update(leaf_group_counts(leaves,len(tasks),initial_not_applicable=False))
  stats['leaf_group_source']='initial four plus actual counterfactual leaves; excludes pseudo training rows and fallback placeholders'
  if cfg.get('ht_weight_cap') is not None:
   cap=cfg['ht_weight_cap'];assert cap==10
   for row in pseudo:row.ucf['ht_weight_cap']=cap
   invs=[1/n['rho'] for n in ns]
   stats['ht_truncated_node_fraction']=sum(x>cap for x in invs)/max(1,len(invs))
   stats['ht_removed_weight_mass_fraction']=sum(max(0.,x-cap) for x in invs)/max(1e-30,sum(invs))
  return first+pseudo,stats,events
 return collect
