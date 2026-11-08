"""Isolated one-token probes. Never modify a branch or execute an environment action."""
import math,time
from cert_entropy import token_map,entropies
from cert_select import stable_seed

KEYS=('generated_tokens','input_tokens','executed_actions','invalid_actions','request_seconds')
def masses(fence_logprobs,action_logprobs,fence_id,mapping):
    def prob(x):
        y=math.exp(float(x.logprob if hasattr(x,'logprob') else x))
        assert math.isfinite(y) and 0<=y<=1+1e-6
        return min(1.,y)
    q=prob(fence_logprobs[fence_id]) if fence_id in fence_logprobs else 0.
    conditional={a:0. for a in range(1,5)}
    for t,lp in action_logprobs.items():
        if int(t) in mapping:conditional[mapping[int(t)]]+=prob(lp)
    assert sum(conditional.values())<=1+1e-5
    p={a:q*v for a,v in conditional.items()};p[0]=max(0.,1-sum(p.values()))
    assert abs(sum(p.values())-1)<1e-5
    return p,dict(format_mass=q,conditional_action_mass=sum(conditional.values()),
      fence_missing_from_returned_top20=fence_id not in fence_logprobs,
      missing_action_token_ids=sorted(set(mapping)-set(map(int,action_logprobs))),
      format_returned_mass=sum(prob(v) for v in fence_logprobs.values()),
      action_returned_mass=sum(prob(v) for v in action_logprobs.values()),
      interpretation='specified fence / first-token proxy; residual includes truncation and other formats')

def probe_nodes(candidates,tasks,engine,tokenizer,cfg,base_seed):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    mapping,_=token_map(tokenizer);fence=tokenizer.encode('```',add_special_tokens=False)
    assert len(fence)==1 and tokenizer.decode(fence)=='```',(fence,'Unsupported fence tokenization')
    backend=engine.llm_engine;serial=getattr(engine,'_patr_request_serial',0)
    pending={};results={};cost=dict.fromkeys(KEYS,0);start=time.monotonic()
    # No rollout requests coexist with probes. Independent seeds and request IDs.
    for ti,ns in candidates.items():
        for n in ns:
            key=(ti,n['j'],n['turn']);results[key]={}
            for kind,suffix in [('format',[]),('action',fence)]:
                ids=list(n['branch'].ids)+suffix
                assert len(ids)<cfg['max_context_tokens'],'Probe context would exceed frozen context limit'
                params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,logprobs=20,
                  seed=stable_seed('cert-probe-v2',base_seed,tasks[ti]['uid'],n['j'],n['turn'],kind),
                  stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY)
                rid=f'cert-probe-{serial}';serial+=1
                backend.add_request(rid,{'prompt_token_ids':ids},params)
                pending[rid]=(key,kind,time.monotonic());cost['input_tokens']+=len(ids)
    while pending:
        if time.time()>=cfg.get('_deadline_unix',float('inf')):
            for rid in pending:backend.abort_request(rid)
            raise TimeoutError('Deadline during isolated probes')
        for result in backend.step():
            if not result.finished:continue
            key,kind,ts=pending.pop(result.request_id);out=result.outputs[0]
            assert len(out.token_ids)==1 and len(out.logprobs)==1
            results[key][kind]={int(t):float(lp.logprob) for t,lp in out.logprobs[0].items()}
            cost['generated_tokens']+=len(out.token_ids);cost['request_seconds']+=time.monotonic()-ts
    engine._patr_request_serial=serial
    for ti,ns in candidates.items():
        for n in ns:
            raw=results[(ti,n['j'],n['turn'])]
            p,d=masses(raw['format'],raw['action'],fence[0],mapping)
            ha,hc,classes=entropies(n['branch'].env,p)
            d.update(format_missing=d['fence_missing_from_returned_top20'],action_first=n.get('observed_format',{}).get('action_first',False),
              action_first_parse_failed=n.get('observed_format',{}).get('action_first',False) and n.get('parent_action') not in (1,2,3,4),fence_token_id=fence[0],raw_probe_logprobs=raw)
            n.update(pi=p,H_act=ha,H_cons=hc,classes=classes,diagnostics=d)
    cost['seconds']=time.monotonic()-start;cost['nodes']=sum(map(len,candidates.values()))
    return cost
