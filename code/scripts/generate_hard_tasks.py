"""CPU dataset build, geometric dedup, exact shortest solutions and SHA receipts."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import patr_core as core
from task_envs import SokobanEnv,generate_flh_map,lake_metadata,map_hash,sokoban_agent,WORDS

def write(p,x):core.atomic_json(p,x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_soko(t):return json.dumps({k:t[k] for k in ['walls','player','box','goal']},sort_keys=True,separators=(',',':'))
def soko_task(seed):
    rng=np.random.default_rng(seed)
    border={(r,c) for r in range(6) for c in range(6) if r in (0,5) or c in (0,5)}
    while True:
        walls=border|{(r,c) for r in range(1,5) for c in range(1,5) if rng.random()<.15}
        free=sorted(set((r,c) for r in range(1,5) for c in range(1,5))-walls)
        if len(free)<3:continue
        player,box,goal=[free[i] for i in rng.choice(len(free),3,replace=False)]
        t=dict(environment='sokoban',seed=seed,size=6,walls=sorted(walls),player=player,box=box,goal=goal)
        e=SokobanEnv(t);solution=e.solution()
        if solution:
            direct=e.manhattan(box,goal)
            t.update(shortest_path_length=len(solution),requires_detour=len(solution)>direct,map_sha256=hashlib.sha256(canonical_soko(t).encode()).hexdigest())
            return t,solution

def build(kind,out,original):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists(),'Dataset already frozen; do not regenerate silently'
    original_data=core.make_datasets();old_tasks=sum(original_data.values(),[])
    for path in original.glob('*/tasks.json'):old_tasks+=json.loads(path.read_text())
    old_seeds={t['seed'] for t in old_tasks};old_hashes=set()
    for t in old_tasks:
        e=core.make_env(t,dict(is_slippery=False,max_turns=10));old_hashes.add(lake_metadata(e)['map_sha256'])
    seen=set(old_hashes) if kind=='flh' else set();used_seeds=set(old_seeds)
    rng=np.random.default_rng(20261107 if kind=='flh' else 20261108)
    candidate=1000000 if kind=='flh' else 2000000
    data={};solutions={};maps={}
    for split,n in [('train',10000),('validation',100),('test',100),('supplemental',500)]:
        rows=[]
        while len(rows)<n:
            candidate+=1
            if candidate in used_seeds:continue
            if kind=='flh':
                size=int(rng.integers(9,13));p=float(rng.uniform(.6,.75))
                board,meta=generate_flh_map(size,p,candidate);solution=meta.pop('solution')
                t=dict(seed=candidate,size=size,p=p,environment='flh',**meta)
            else:t,solution=soko_task(candidate)
            if t['map_sha256'] in seen:continue
            seen.add(t['map_sha256']);used_seeds.add(candidate)
            t.update(uid=f'{kind}_{candidate}',split=split);rows.append(t);solutions[t['uid']]=solution
            maps[t['uid']]=board if kind=='flh' else {k:t[k] for k in ['walls','player','box','goal']}
        data[split]=rows;print(kind,split,len(rows),flush=True)
    write(out/'data.json',data)
    write(out/'maps.json',maps)
    write(out/'supplemental500/tasks.json',data['supplemental'])
    write(out/'supplemental500/protocol.json',dict(task_file_sha256=sha(out/'supplemental500/tasks.json'),evaluation_seed=200042))
    if kind=='sokoban':
        # Training-only first 1000 unique maps. Evaluation solutions are never teachers.
        with (out/'sft_messages.jsonl').open('w') as f:
            for t in data['train'][:1000]:
                e=SokobanEnv(t);a=sokoban_agent(10);a.update_from_env(e.render(),0,False,{})
                for action in solutions[t['uid']]:
                    a.update_from_model('```'+WORDS[action]+'```');obs,r,done,info=e.step(action)
                    if not done:a.update_from_env(obs,r,done,info)
                assert e.success()
                f.write(json.dumps(dict(task_uid=t['uid'],messages=a.chat_completions))+'\n')
    lengths={s:dict(n=len(ts),min=min(t['shortest_path_length'] for t in ts),max=max(t['shortest_path_length'] for t in ts),detours=sum(t['requires_detour'] for t in ts)) for s,ts in data.items()}
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'manifest.json',dict(environment=kind,files=files,split_statistics=lengths,seed_overlap_with_original=0,map_overlap_with_original=0,unique_maps=sum(map(len,data.values())),geometry_dedup=True,generator_sha256=sha(Path(__file__)),task_envs_sha256=sha(Path(__file__).resolve().parents[1]/'task_envs.py'),sokoban_generation='6x6 border walls; independent internal walls p=.15; uniform distinct player/box/goal; BFS rejection to 1..10 solution steps' if kind=='sokoban' else None,detour_definition='shortest solution > initial box-goal Manhattan' if kind=='sokoban' else 'shortest path > start-goal Manhattan',new_eval_sets=['main100','supplemental500'],legacy_filtered409_map8='not transplanted across environments'))
    # Reconstruct every record independently, replay all solutions and check frozen metadata.
    for split,tasks in data.items():
        for t in tasks:
            if kind=='flh':
                board,meta=generate_flh_map(t['size'],t['p'],t['seed'])
                assert meta['map_sha256']==t['map_sha256'] and meta['shortest_path_length']==t['shortest_path_length']
            else:
                e=SokobanEnv(t)
                for a in e.solution():e.step(a)
                assert e.success()
    write(out/'cpu_acceptance.json',dict(passed=True,tasks=sum(map(len,data.values())),golden_replay=1000 if kind=='sokoban' else None,sha256=sha(out/'manifest.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['flh','sokoban'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--original-eval',type=Path,default=core.ROOT/'eval_sets');a=p.parse_args();build(a.environment,a.output,a.original_eval)
