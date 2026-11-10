"""UCF-v2.2 / v2.2-ff adapter. The original v2 module remains byte-identical."""
import copy,gzip,json,math,os,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from cert_collect import initial,replay,stream
from cert_select import stable_seed
from tokens import fence_id
from cert_entropy import token_map
from ucf_probe import probe_exact,requests,forced_logprobs
from ucf_select import Controller,feature
from ucf_corrected import complement_stratum, sampling_restriction, first_complement_output, token_in_complement, leave_question_out_constants, select_corrected
from ucf_weights import fork_rows,generation_end

def initial_ucf(core,tasks,engine,tokenizer,cfg,seed):
 first,nodes,cost,_=initial(core,tasks,engine,tokenizer,cfg,seed,topk=1,with_probes=False)
 from first_fence import locate_token_prefix
 candidates={ti:[] for ti in range(len(tasks))}
 by_branch={(b.task_id,int(b.lineage)):b for b in first}
 fence=fence_id(tokenizer);coverage=dict(turns=len(nodes),turns_with_fence=0,candidates=0,prefix_budget_exhausted=0,prefix_already_closed=0,SFT_nonopening_fence=0)
 for n in nodes:
  o=by_branch[(n['branch'].task_id,n['j'])];start=n['boundary'];end=generation_end(o,start)
  bd=locate_token_prefix(o.ids,start,end,tokenizer)
  if bd is None:continue
  coverage['turns_with_fence']+=1
  prefix_text=tokenizer.decode(o.ids[start:bd],skip_special_tokens=True)
  if cfg.get('fork_point')!='first_fence' and prefix_text[:prefix_text.find('```')].strip():
   coverage['SFT_nonopening_fence']+=1;continue
  if cfg.get('response_protocol')=='first_complete_action_block':
   from response_protocol import first_block
   if first_block(tokenizer.decode(o.ids[start:bd],skip_special_tokens=True)):
    coverage['prefix_already_closed']+=1;continue
  parent=copy.deepcopy(n['branch'])
  for attr in ['ids','mask','old_logprobs']:setattr(parent,attr,list(getattr(o,attr)[:bd]))
  rem=min(cfg['max_response_tokens']-parent.ntokens,cfg['max_context_tokens']-bd)
  if rem<=0:
   coverage['prefix_budget_exhausted']+=1;continue
  # No parser/finish_reason/action test affects candidacy. This assertion is
  # an evidence integrity check, not exclusion based on the sampled suffix.
  assert bd<end, 'Causal node has budget but the original successor is missing'
  n.update(branch=parent,boundary=bd,turn_start=start,fork_point='first_fence')
  candidates[parent.task_id].append(n);coverage['candidates']+=1
 probe=probe_first_fence(candidates,engine,tokenizer,cfg,seed)
 probe['coverage']=coverage
 return first,candidates,cost,probe


def probe_first_fence(candidates,engine,tokenizer,cfg,seed):
 from vllm import SamplingParams
 from first_fence import action_masses
 from cert_entropy import entropies
 mapping,_=token_map(tokenizer);assert len(mapping)==8
 unique={}
 for ns in candidates.values():
  for n in ns:unique.setdefault(tuple(n['branch'].ids),[]).append(n)
 prefixes=list(unique);jobs=[];raw={i:{} for i in range(len(prefixes))}
 for i,prefix in enumerate(prefixes):
  assert len(prefix)+1<cfg['max_context_tokens'], 'Prompt scoring must fit context; no silent candidate removal'
  for x in mapping:
   jobs.append(((i,x),list(prefix)+[x],SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,prompt_logprobs=1,seed=stable_seed(seed,'v21-score',i,x))))
 total=dict(input_tokens=0,generated_tokens=0,request_seconds=0.,seconds=0.)
 # Bounded returned prompt-logprob memory. No rollout requests coexist.
 for start in range(0,len(jobs),32):
  result,cost=requests(engine,jobs[start:start+32])
  for k in total:total[k]+=cost[k]
  for (i,x),out in result.items():
   assert out.prompt_logprobs is not None
   value=float(out.prompt_logprobs[len(prefixes[i])][x].logprob)
   assert math.isfinite(value)
   raw[i][x]=value
 for i,prefix in enumerate(prefixes):
  p=action_masses(raw[i],mapping)
  for n in unique[prefix]:
   ha,hc,classes=entropies(n['branch'].env,p)
   n.update(pi=p.copy(),H_act=ha,H_cons=hc,classes=classes,exact_logprobs=raw[i].copy(),
            action_prompt_logprobs=raw[i].copy(),probability_condition='actual first-fence prefix; no fence probability factor')
 total.update(unique_prefixes=len(prefixes),candidate_occurrences=sum(map(len,candidates.values())),scoring_method='vLLM unrestricted prompt_logprobs for all eight action-token variants')
 return total

def counterfactual(core,picked,first,tasks,engine,tokenizer,cfg,seed,step):
 from vllm import SamplingParams
 mapping,_=token_map(tokenizer);fence=fence_id(tokenizer);orig={(b.task_id,int(b.lineage)):b for b in first};children={};jobs=[];node_map={};rows=[]
 for ti,ns in picked.items():
  for n in ns:
   o=orig[(ti,n['j'])];bd=n['boundary'];end=generation_end(o,bd)
   s=complement_stratum(n,o.ids[bd],mapping);n['stratum']=s
   s['parse_mismatch']=False
   n['original_parse_mismatch']=s['first_action']!=int(o.agent._parse_model_response(o.trace[n['turn']-1]['response'])[1])
   key=(ti,n['j'],n['turn']);node_map[key]=n
   if s['cf_fallback']:continue
   b=replay(core,n,tasks[ti],cfg);b.lineage=f"{n['j']}@{n['turn']}.cf";children[key]=b
   rem=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
   assert rem>=1,'No action-token budget at a causal node'
   jobs.append((key,list(b.ids),SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,**sampling_restriction(s),logprobs=1,seed=stable_seed(seed,tasks[ti]['uid'],n['j'],n['turn'],step,'cf'))))
 forced={};fcost=dict(input_tokens=0,generated_tokens=0,request_seconds=0.,seconds=0.)
 for offset in range(0,len(jobs),4):
  batch,charge=requests(engine,jobs[offset:offset+4]);forced.update(batch)
  for k in fcost:fcost[k]+=charge[k]
 tail_jobs=[];rejection_rows=[];selected_outputs={};exhausted=[];residual_expected=[];residual_observed=[]
 for key,b in list(children.items()):
  n=node_map[key]
  from cert_entropy import consequence_classes
  classes=consequence_classes(b.env)
  selected,audit=first_complement_output(forced[key].outputs,n['stratum'],mapping,classes)
  rejection_rows.append(dict(key=key,seed=stable_seed(seed,tasks[b.task_id]['uid'],n['j'],n['turn'],step,'cf'),pi_o=n['stratum']['pi_o'],pi_cf=n['stratum']['pi_cf'],original_class=n['stratum']['original_class'],chosen_index=None if selected is None else selected.index,exhausted=selected is None,proposals=audit))
  n['rejection_proposals']=len(audit);n['rejection_chosen_index']=None if selected is None else selected.index
  if selected is None:
   n['stratum'].update(cf_fallback=True,fallback_reason='all_256_proposals_in_original_class');exhausted.append(key);del children[key];continue
  selected_outputs[key]=selected;x=int(selected.token_ids[0]);n['forced_token']=x
  assert token_in_complement(x,n['stratum'],mapping,classes)
  expected=n['pi'][0]/n['stratum']['pi_cf'] if n['stratum']['original_class']!=classes[0] else 0.
  residual_expected.append(expected);residual_observed.append(x not in mapping)
  ids=b.ids+[x]
  rem=min(cfg['max_response_tokens']-b.ntokens-1,cfg['max_context_tokens']-len(ids))
  if rem==0 or x==tokenizer.eos_token_id:continue
  tail_jobs.append((key,ids,SamplingParams(temperature=cfg['temperature'],top_p=1.,top_k=-1,max_tokens=rem,logprobs=1,seed=stable_seed(seed,tasks[b.task_id]['uid'],b.lineage,b.turns),stop_token_ids=[tokenizer.eos_token_id])))
 # Score every sampled X in the unrestricted distribution, including residual tokens.
 score_jobs=[(key,list(node_map[key]['branch'].ids)+[node_map[key]['forced_token']],SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,prompt_logprobs=1,seed=stable_seed(seed,'cf-raw-score',*key))) for key in children]
 scored,scost=requests(engine,score_jobs)
 for key in children:
  n=node_map[key];x=n['forced_token'];n['forced_raw_lp']=float(scored[key].prompt_logprobs[len(n['branch'].ids)][x].logprob)
  assert math.isfinite(n['forced_raw_lp'])
 tails,tcost=requests(engine,tail_jobs);active=[];done={};mismatch=0
 for key,b in children.items():
  n=node_map[key];x=n['forced_token'];lp=n['forced_raw_lp']
  restricted=float(selected_outputs[key].logprobs[0][x].logprob)
  allowed_mass=n['stratum']['pi_cf']
  n['prompt_vs_proposal_logprob_difference']=(lp-math.log(allowed_mass) if n['stratum']['allowed_tokens'] is not None else lp)-restricted
  n['forced_vllm_logprob']=restricted
  n['proposal_was_restricted']=n['stratum']['allowed_tokens'] is not None
  n['forced_old_logprob_unrestricted']=lp
  tail=tails[key].outputs[0] if key in tails else SimpleNamespace(token_ids=[],logprobs=[],finish_reason='stop' if x==tokenizer.eos_token_id else 'length')
  ids=[x]+list(tail.token_ids)
  # Token IDs / old logprobs of reasoning and first fence were copied into b.
  # The unchanged parser receives the complete current-turn response text.
  text_ids=b.ids[n['turn_start']:]+ids
  out=SimpleNamespace(token_ids=ids,logprobs=[{x:SimpleNamespace(logprob=lp)}]+list(tail.logprobs),text=tokenizer.decode(text_ids,skip_special_tokens=True),finish_reason=tail.finish_reason)
  core.apply_response(b,out,tokenizer,cfg)
  parsed=int(b.trace[-1].get('action') or 0) if b.trace else 0
  n['cf_parse_mismatch']=parsed!=mapping.get(x,0);mismatch+=n['cf_parse_mismatch']
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
 audit_path=Path(cfg['ucf_log_dir'])/f'rejection_{step:04d}.jsonl.gz' if cfg.get('ucf_log_dir') else None
 if audit_path:
  with gzip.open(audit_path,'xt') as f:
   for row in rejection_rows:f.write(json.dumps(row)+'\n')
 expected_count=sum(residual_expected);variance=sum(p*(1-p) for p in residual_expected);observed_count=sum(residual_observed)
 z=abs(observed_count-expected_count)/math.sqrt(variance) if variance>0 else (0. if observed_count==expected_count else float('inf'))
 rejection=dict(proposal_count=sum(len(r['proposals']) for r in rejection_rows),nonSTAY_batches=sum(len(r['proposals'])==256 for r in rejection_rows),nonSTAY_accepted=sum(len(r['proposals'])==256 and not r['exhausted'] for r in rejection_rows),nonSTAY_fallback_fraction=len(exhausted)/max(1,sum(len(r['proposals'])==256 for r in rejection_rows)),exhausted_batches=len(exhausted),fallback_fraction=len(exhausted)/max(1,len(rejection_rows)),accepted=len(children),forbidden_accepted=0,residual_observed=observed_count,residual_expected=expected_count,residual_variance=variance,residual_z=z,audit_path=str(audit_path) if audit_path else None)
 cost=dict(forced=fcost,scoring=scost,tail=tcost,continuation=continuation_cost,cf_parse_mismatch=mismatch/max(1,len(children)),cf_count=len(children),rejection=rejection)
 return rows,done,cost

def make_collect(core,original):
 def collect(tasks,engine,tokenizer,cfg,method,rng,evaluation=False):
  if evaluation:return original(tasks,engine,tokenizer,cfg,method,rng,evaluation=True)
  assert cfg['temperature']==1. and cfg['initial_branches']==4
  from ucf_solvability import pilot_enabled,prepare_pilot,pilot_statistics,controller_bootstrap_c
  pilot=pilot_enabled(cfg)
  out=Path(cfg['ucf_log_dir']);out.mkdir(parents=True,exist_ok=True)
  step=cfg.get('_step',0)
  progress=os.environ.get('PATR_PROGRESS_PATH')
  if progress and (Path(progress).parent/'status.json').exists():step=json.loads((Path(progress).parent/'status.json').read_text()).get('current_step',step)
  state=out/'controller.json'
  data=json.loads(state.read_text()) if state.exists() else dict(c=cfg['ucf_c0'],history=[],last_step=-1)
  assert step>data['last_step'],'Refuse ambiguous resume: reconcile controller with checkpoint before retry'
  ctrl=Controller(controller_bootstrap_c(data['c']) if pilot else data['c'],data['history']);c_step=ctrl.c
  seed=int(rng.integers(0,2**31-1))
  first,candidates,icost,pcost=initial_ucf(core,tasks,engine,tokenizer,cfg,seed)
  # Part B begins immediately after the first fence; parser disagreement is recorded, never selected on.
  mapping,_=token_map(tokenizer);fence=fence_id(tokenizer)
  originals={(b.task_id,int(b.lineage)):b for b in first}
  for ti,ns in candidates.items():
   for n in ns:
    original_branch=originals[(ti,n['j'])];bd=n['boundary'];end=generation_end(original_branch,bd)
    st=complement_stratum(n,original_branch.ids[bd],mapping)
    st['parse_mismatch']=False
    n['original_parse_mismatch']=st['first_action']!=int(original_branch.agent._parse_model_response(original_branch.trace[n['turn']-1]['response'])[1])
    n['stratum']=st
  if pilot:
   constants,budget=prepare_pilot(candidates,first,cfg,target=2.)
  else:
   task_weights={ti:[feature(n,first,cfg['ucf_variant'])[0] for n in ns] for ti,ns in candidates.items()}
   constants=leave_question_out_constants(task_weights,target=2.) if cfg.get('ucf_controller','v2')=='v2' else {ti:c_step for ti in candidates}
   budget=dict(target=2.,calibration='leave_one_question_out',c_by_task=constants,
               expected_forks=sum(sum(min(1.,max(.03,constants[ti]*w)) for w in ws) for ti,ws in task_weights.items())/len(tasks),
               exact_batch_budget_guaranteed=False,
               note='Held-out question support cannot tune its own c; realized/expected held-out budget may differ from 2')
  c_step=sum(constants.values())/len(constants)  # logging only; nodes use their question's c
  core.atomic_json(out/f'budget_{step:04d}.json',budget)
  picked,selection=select_corrected(candidates,first,tasks,constants,seed,step,out/f'select_{step:04d}.jsonl',cfg['ucf_variant'],pilot=pilot)
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
  stats['conditioned_fence_tokens']=0
  stats['copied_fence_tokens']=ccost['cf_count']
  stats['fork_point']='first_fence'
  stats['ucf_version']='UCF-v2.2-ff' if cfg.get('fork_point')=='first_fence' else 'UCF-v2.2'
  if pilot:
   stats['pilot_solvability']=pilot_statistics(candidates,first,children,budget)
   stats['pilot_arm']='P2' if cfg.get('early_exploration_floor',False) else 'P1'
   stats['ucf_version']='UCF-v2.2+solvability-gate'+('+early-floor' if cfg.get('early_exploration_floor',False) else '')
  stats['rejection_sampling']=ccost['rejection']
  if cfg.get('response_protocol')=='first_complete_action_block':
   from response_protocol import protocol_metrics
   stats['counterfactual_protocol_metrics']=protocol_metrics(children.values())
  stats['response_protocol']=cfg.get('response_protocol','legacy_last_block')
  stats['first_fence_coverage']=pcost['coverage']
  stats['candidate_coverage']=pcost['coverage']['candidates']/max(1,pcost['coverage']['turns'])
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
   original_parse_mismatch_fraction=sum(n['original_parse_mismatch'] for n in all_nodes)/max(1,len(all_nodes)),
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
