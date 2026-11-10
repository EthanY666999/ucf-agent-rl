"""Untruncated action masses through unrestricted prompt scoring."""
import math,os,time
from cert_entropy import token_map,entropies
from tokens import fence_id
from cert_select import stable_seed

def requests(engine,jobs):
 backend=engine.llm_engine;serial=getattr(engine,'_patr_request_serial',0);pending={};result={};cost=dict(input_tokens=0,generated_tokens=0,request_seconds=0.)
 start=time.monotonic();last_progress=0.
 for key,ids,params in jobs:
  rid=f'ucf-{serial}';serial+=1;backend.add_request(rid,{'prompt_token_ids':ids},params);pending[rid]=(key,time.monotonic());cost['input_tokens']+=len(ids)
 engine._patr_request_serial=serial
 while pending:
  if time.monotonic()-last_progress>=5 and os.environ.get('PATR_PROGRESS_PATH'):
   from patr_core import atomic_json
   atomic_json(os.environ['PATR_PROGRESS_PATH'],dict(phase='ucf_probe_or_forced_scoring',active=len(pending),completed=len(result),generated_tokens=cost['generated_tokens'],updated_unix=time.time()))
   last_progress=time.monotonic()
  for out in backend.step():
   if not out.finished:continue
   key,ts=pending.pop(out.request_id);result[key]=out;cost['generated_tokens']+=sum(len(x.token_ids) for x in out.outputs);cost['request_seconds']+=time.monotonic()-ts
 cost['seconds']=time.monotonic()-start
 return result,cost

def forced_logprobs(node,fence,x):
 assert isinstance(fence,int) and fence>=0
 a,b=node['exact_logprobs'][x]
 assert math.isfinite(a) and math.isfinite(b)
 return a,b

def actor_logprobs(actor,prefixes,mapping,pad_id=0,batch_size=8,token_limit=12000,fence=None):
 """One no-grad actor prefill per prefix + fence; full-vocabulary normalizers.

 Hidden state at L-1 predicts the fence; state at L predicts action tokens.
 Match the frozen trainer's CUDA bf16 autocast, with fp32 log_softmax.
 """
 import torch
 from contextlib import nullcontext
 assert fence is not None,'Pass the tokenizer-derived fence id'
 keys=list(mapping);assert len(keys)==len(set(keys))
 device=next(actor.parameters()).device;was_training=actor.training;actor.eval();results={}
 order=sorted(range(len(prefixes)),key=lambda i:len(prefixes[i]));chunks=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=batch_size or (len(prefixes[i])+1)*(len(chunk)+1)>token_limit):chunks.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:chunks.append(chunk)
 try:
  with torch.no_grad():
   for chunk in chunks:
    lengths=[len(prefixes[i]) for i in chunk];assert min(lengths)>0
    ids=torch.full((len(chunk),max(lengths)+1),pad_id,dtype=torch.long,device=device);attention=torch.zeros_like(ids)
    for row,i in enumerate(chunk):
     seq=list(prefixes[i])+[fence];ids[row,:len(seq)]=torch.tensor(seq,device=device);attention[row,:len(seq)]=1
    ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with ctx:
     hidden=actor.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
     idx=torch.arange(len(chunk),device=device);lens=torch.tensor(lengths,device=device)
     final=torch.stack([hidden[idx,lens-1],hidden[idx,lens]],dim=1)
     logits=actor.lm_head(final)
    lp=logits.float().log_softmax(-1)
    for row,i in enumerate(chunk):
     f=float(lp[row,0,fence]);results[i]={x:(f,float(lp[row,1,x])) for x in keys}
    del ids,attention,hidden,final,logits,lp
 finally:actor.train(was_training)
 return [results[i] for i in range(len(prefixes))]

def probe_exact(candidates,engine,tokenizer,cfg,base_seed):
 mapping,_=token_map(tokenizer);assert len(mapping)==8
 fence=fence_id(tokenizer)
 unique={};started=time.monotonic()
 for ti,ns in candidates.items():
  for n in ns:unique.setdefault(tuple(n['branch'].ids),[]).append(n)
 prefixes=list(unique)
 assert all(len(p)+1<cfg['max_context_tokens'] for p in prefixes)
 scored=actor_logprobs(engine._ucf_actor,prefixes,mapping,tokenizer.pad_token_id,batch_size=cfg.get('actor_probe_batch_size',8),token_limit=cfg.get('actor_probe_token_limit',12000),fence=fence)
 diffs=[]
 for prefix,raw in zip(prefixes,scored):
  p={a:sum(math.exp(raw[x][0]+raw[x][1]) for x in mapping if mapping[x]==a) for a in range(1,5)}
  assert sum(p.values())<=1+1e-6;p[0]=max(0.,1-sum(p.values()))
  for n in unique[prefix]:
   ha,hc,classes=entropies(n['branch'].env,p)
   observed=n.get('vllm_observed',{})
   delta={}
   if 'fence' in observed:delta['fence']=next(iter(raw.values()))[0]-observed['fence']
   if observed.get('action_token') in raw:delta['action']=raw[observed['action_token']][1]-observed['action']
   diffs.extend(delta.values())
   n.update(pi=p.copy(),H_act=ha,H_cons=hc,classes=classes,exact_logprobs=raw.copy(),actor_vllm_difference=delta)
 elapsed=time.monotonic()-started
 return dict(input_tokens=sum(len(p)+1 for p in prefixes),generated_tokens=0,request_seconds=elapsed,seconds=elapsed,unique_prefixes=len(unique),candidate_occurrences=sum(map(len,candidates.values())),scoring_method='actor no_grad prefix+fence, full-vocabulary fp32 log_softmax; training bf16 autocast',actor_vllm_logprob_differences=diffs,comparison_coverage=len(diffs),comparison_source='existing unrestricted initial rollout sampled-token logprobs only; no extra vLLM scoring requests')
