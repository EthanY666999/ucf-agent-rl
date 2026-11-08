"""CERT adapter. Sample IDs and prefix logprobs are never retokenized/recomputed."""
import copy,json,math,os,time
from collections import Counter
from cert_entropy import token_map,distribution,entropies
from cert_select import deduplicate,select,stable_seed


def replay(core,node,task,cfg):
    parent=node['branch'];env=core.make_env(task,cfg)
    for a in parent.actions:
        _,_,done,_=env.step(a)
        assert not done,'Recorded pre-action prefix is terminal'
    assert int(env.s)==node['state'] and (env.desc==parent.env.desc).all(),'Replay state mismatch'
    if hasattr(env,'anchor'):assert env.anchor()==parent.env.anchor(), 'Replay box/player mismatch'
    child=copy.deepcopy(parent);child.env=env
    child.lineage=f"{node['j']}@{node['turn']}.0"
    child.order=(4+node['j']*cfg['max_turns']+node['turn'],)
    for key in ('ids','mask','old_logprobs'):
        assert getattr(child,key)==getattr(parent,key)
    assert len(child.ids)==node['boundary'] and child.turns==node['turn']-1
    return child


def stream(core,branches,tasks,engine,tokenizer,cfg,base_seed,topk=20,capture=False,mapping=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    backend=engine.llm_engine;serial=getattr(engine,'_patr_request_serial',0)
    pending={};done=[];nodes=[];costs={};started=time.monotonic();last_progress=0
    def enqueue(b):
        nonlocal serial
        rem=min(cfg['max_response_tokens']-b.ntokens,cfg['max_context_tokens']-len(b.ids))
        if rem<=0:
            b.terminal='context_limit' if len(b.ids)>=cfg['max_context_tokens'] else 'token_limit';done.append(b);return
        n=None
        if capture:
            snapshot=copy.deepcopy(b)
            d=core.distance(b.env)
            n=dict(branch=snapshot,j=int(b.lineage),turn=b.turns+1,state=int(b.env.s),boundary=len(b.ids),
                   S=0. if d is None else max(0.,min(1.,1-d/max(1,b.initial_distance if getattr(b.env,"environment",None)=="sokoban" else b.env.nrow+b.env.ncol-2))))
        params=SamplingParams(temperature=cfg['temperature'],top_p=1.,top_k=-1,max_tokens=rem,logprobs=topk,
             seed=stable_seed(base_seed,tasks[b.task_id]['uid'],b.lineage,b.turns),
             stop_token_ids=[tokenizer.eos_token_id],output_kind=RequestOutputKind.FINAL_ONLY)
        rid=f'cert-stream-{serial}';serial+=1
        key=(b.task_id,b.lineage);costs.setdefault(key,dict(generated_tokens=0,input_tokens=0,executed_actions=0,invalid_actions=0,request_seconds=0.))
        costs[key]['input_tokens']+=len(b.ids)
        backend.add_request(rid,{'prompt_token_ids':list(b.ids)},params)
        pending[rid]=(b,n,time.monotonic())
    for b in branches:enqueue(b)
    while pending:
        now=time.monotonic()
        deadline=cfg.get('_deadline_unix',float('inf'))
        if time.time()>=deadline:
            for rid in pending:backend.abort_request(rid)
            raise TimeoutError('Compute deadline during CERT sampling; no partial cell accepted')
        if now-last_progress>=5 and os.environ.get('PATR_PROGRESS_PATH'):
            core.atomic_json(os.environ['PATR_PROGRESS_PATH'],dict(phase='cert_initial' if capture else 'cert_continuations',active=len(pending),completed=len(done),updated_unix=time.time()))
            last_progress=now
        for result in backend.step():
            if not result.finished:continue
            b,n,ts=pending.pop(result.request_id);out=result.outputs[0];c=costs[(b.task_id,b.lineage)]
            c['request_seconds']+=time.monotonic()-ts;c['generated_tokens']+=len(out.token_ids)
            before=b.turns
            if n is not None:
                p,diagnostics=distribution(out,tokenizer,mapping)
                ha,hc,classes=entropies(n['branch'].env,p)
                n.update(pi=p,H_act=ha,H_cons=hc,classes=classes,diagnostics=diagnostics)
            # Frozen V1 operation: sampled IDs untouched, environment text only encoded.
            core.apply_response(b,out,tokenizer,cfg)
            if b.turns>before:
                c['executed_actions']+=1;c['invalid_actions']+=int(b.actions[-1]==0)
            if n is not None:
                n['parent_action']=b.actions[-1] if b.turns>before else None
                n['diagnostics']['action_first_parse_failed']=n['diagnostics']['action_first'] and n['parent_action'] not in (1,2,3,4)
                nodes.append(n)
            if not b.terminal and len(b.actions)>=cfg['repeat_limit'] and len(set(b.actions[-cfg['repeat_limit']:]))==1:b.terminal='pruned_repeat'
            if b.terminal:done.append(b)
            else:enqueue(b)
    engine._patr_request_serial=serial
    done.sort(key=lambda b:(b.task_id,b.order))
    total={k:sum(c[k] for c in costs.values()) for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions','request_seconds')}
    total['seconds']=time.monotonic()-started
    return done,nodes,total,costs


def initial(core,tasks,engine,tokenizer,cfg,base_seed,topk=1,with_probes=True):
    assert topk==1,'Production logprobs must remain 1'
    # Execute frozen collect / SamplingParams / apply_response, with routing beyond H.
    # The observer copies pre-response state only; it consumes no RNG.
    nodes=[];original=core.apply_response;mapping,_=token_map(tokenizer)
    def observe(b,out,tok,c):
        snapshot=copy.deepcopy(b);d=core.distance(b.env)
        n=dict(branch=snapshot,j=int(b.lineage),turn=b.turns+1,state=int(b.env.s),boundary=len(b.ids),
               S=0. if d is None else max(0.,min(1.,1-d/max(1,b.initial_distance if getattr(b.env,"environment",None)=="sokoban" else b.env.nrow+b.env.ncol-2))))
        before=b.turns;original(b,out,tok,c)
        n['parent_action']=b.actions[-1] if b.turns>before else None
        _,observed=distribution(out,tok,mapping)
        n['observed_format']=observed
        nodes.append(n)
    class FixedSeed:
        def integers(self,*args,**kwargs):return base_seed
    c=dict(cfg,branch_interval=cfg['max_turns']+1)
    ts=time.monotonic();core.apply_response=observe
    try:first,stats,_=core.collect(tasks,engine,tokenizer,c,'patr',FixedSeed())
    finally:core.apply_response=original
    cost={k:stats[k] for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions')}
    cost.update(seconds=time.monotonic()-ts,request_seconds=0.,request_seconds_unavailable_for_frozen_initial=True)
    if with_probes:
        from cert_probe import probe_nodes
        candidates,_=prepare_nodes(nodes,first,tasks)
        pcost=probe_nodes(candidates,tasks,engine,tokenizer,cfg,base_seed)
        # Probe only unique prefixes, then retain canonical nodes.
        nodes=[n for ns in candidates.values() for n in ns]
        cost['probe_cost']=pcost
        for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions','request_seconds','seconds'):cost[k]+=pcost[k]
    return first,nodes,cost,{}


def prepare_nodes(nodes,initial_branches,tasks):
    candidates={i:deduplicate([n for n in nodes if n['branch'].task_id==i]) for i in range(len(tasks))}
    p_hat={i:sum(b.terminal=='success' for b in initial_branches if b.task_id==i)/4 for i in range(len(tasks))}
    return candidates,p_hat


def continuations(core,candidates,tasks,engine,tokenizer,cfg,base_seed):
    children=[];skips=[];node_by_key={}
    for ti,nodes in candidates.items():
        for n in nodes:
            try:b=replay(core,n,tasks[ti],cfg)
            except AssertionError as e:
                skips.append(dict(uid=tasks[ti]['uid'],j=n['j'],turn=n['turn'],reason=str(e)));continue
            children.append(b);node_by_key[(ti,b.lineage)]=n
    done,_,total,costs=stream(core,children,tasks,engine,tokenizer,cfg,base_seed,1,False)
    by_node={}
    for b in done:
        n=node_by_key[(b.task_id,b.lineage)];boundary=n['boundary']
        for key in ('ids','mask','old_logprobs'):assert getattr(b,key)[:boundary]==getattr(n['branch'],key)
        first=b.actions[n['turn']-1] if len(b.actions)>=n['turn'] else None
        by_node[(b.task_id,n['j'],n['turn'])]=(b,costs[(b.task_id,b.lineage)],first==n['parent_action'] if first is not None else None)
    return by_node,total,skips


def node_record(n):
    return {k:v for k,v in n.items() if k!='branch'}


def make_collect(core,original):
    def collect(tasks,engine,tokenizer,cfg,method,rng,evaluation=False):
        if evaluation:return original(tasks,engine,tokenizer,cfg,method,rng,evaluation=True)
        assert cfg['cert_rounds']==1 and cfg['logprobs_topk']==1 and not cfg['score_prune']
        seed=int(rng.integers(0,2**31-1))
        first,nodes,icost,_=initial(core,tasks,engine,tokenizer,cfg,seed)
        candidates,ph=prepare_nodes(nodes,first,tasks)
        # Step affects only tie-breaking and is recovered from the training status.
        step=cfg.get('_step',0)
        progress=os.environ.get('PATR_PROGRESS_PATH')
        if progress:
            status=__import__('pathlib').Path(progress).parent/'status.json'
            if status.exists():step=json.loads(status.read_text()).get('current_step',step)
        picked={ti:select(ns,cfg['cert_selector'],cfg['cert_forks'],ph[ti],seed,tasks[ti]['uid'],step) for ti,ns in candidates.items()}
        added,ccost,skips=continuations(core,picked,tasks,engine,tokenizer,cfg,seed)
        leaves=first+[x[0] for x in added.values()];leaves.sort(key=lambda b:(b.task_id,b.order))
        groups=[[b for b in leaves if b.task_id==ti] for ti in range(len(tasks))]
        assert all(4<=len(bs)<=4+cfg['cert_forks'] for bs in groups)
        mixed=sum(len({b.reward>0 for b in bs})>1 for bs in groups)
        stats={k:icost[k]+ccost[k] for k in ('generated_tokens','input_tokens','executed_actions','invalid_actions')}
        dup=[v[2] for v in added.values() if v[2] is not None]
        stats.update(probe_cost=icost['probe_cost'],episodes=len(leaves),success_rate=sum(b.reward>0 for b in leaves)/len(leaves),
          invalid_action_fraction=stats['invalid_actions']/stats['executed_actions'] if stats['executed_actions'] else 0.,
          mean_response_tokens=sum(b.ntokens for b in leaves)/len(leaves),mean_turns=sum(b.turns for b in leaves)/len(leaves),
          terminal_counts=dict(Counter(b.terminal for b in leaves)),mixed_groups=mixed,task_groups=len(tasks),mixed_group_fraction=mixed/len(tasks),
          leaf_cap=4+cfg['cert_forks'],max_leaves=max(map(len,groups)),branch_events=len(added),score_prunes=0,
          p_hat_distribution=dict(Counter(ph.values())),fork_turns=dict(Counter(n['turn'] for ns in picked.values() for n in ns)),
          dup_first_action_rate=sum(dup)/len(dup) if dup else None,replay_skips=skips,
          sampled_surprisal=-sum(lp for b in leaves for lp,m in zip(b.old_logprobs,b.mask) if m)/sum(b.ntokens for b in leaves),
          candidate_shortfalls=sum(len(ns)<cfg['cert_forks'] for ns in candidates.values()))
        events=[dict(kind='cert_selection',task_id=ti,p_hat=ph[ti],candidates=[node_record(n) for n in ns],selected=[[n['j'],n['turn']] for n in picked[ti]]) for ti,ns in candidates.items()]
        return leaves,stats,events
    return collect
