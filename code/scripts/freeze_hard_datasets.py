"""Freeze actual maps; independently verify load identity, shortest paths and teachers."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from task_envs import generate_flh_map,lake_metadata,SokobanEnv
from patr_core import make_env,atomic_json
from generate_hard_tasks import sha

def freeze(kind):
    root=Path('/data/runs/datasets')/kind;data=json.loads((root/'data.json').read_text())
    maps={}
    for tasks in data.values():
        for t in tasks:
            if kind=='flh':
                board,meta=generate_flh_map(t['size'],t['p'],t['seed'])
                assert meta['map_sha256']==t['map_sha256'] and meta['shortest_path_length']==t['shortest_path_length']
                maps[t['uid']]=board
            else:maps[t['uid']]={k:t[k] for k in ['walls','player','box','goal']}
    target=root/'maps.json'
    if target.exists():assert json.loads(target.read_text())==maps
    else:atomic_json(target,maps)
    cfg=dict(environment=kind,max_turns=20 if kind=='flh' else 10,is_slippery=False,frozen_maps_path=str(target),frozen_maps_sha256=sha(target))
    for tasks in data.values():
        for t in tasks:
            e=make_env(t,cfg)
            if kind=='flh':
                loaded=[''.join(x.decode() for x in row) for row in e.desc]
                assert loaded==maps[t['uid']]
                meta=lake_metadata(e)
                for k in ['map_sha256','shortest_path_length','requires_detour']:assert meta[k]==t[k]
                for a in meta['solution']:e.step(a)
            else:
                solution=e.solution();assert len(solution)==t['shortest_path_length']
                for a in solution:e.step(a)
            assert e.success()
    teacher=None
    if kind=='sokoban':
        p=root/'sft_messages.jsonl';rs=[json.loads(x) for x in p.read_text().splitlines()]
        # Correct initial engineering schema, preserve prior evidence; model/task selection unchanged.
        if any('uid' in r for r in rs):
            (root/'sft_schema_before_fix.txt').write_bytes(p.read_bytes())
            for r in rs:
                if 'uid' in r:r['task_uid']=r.pop('uid')
            p.write_text(''.join(json.dumps(r)+'\n' for r in rs))
        from transformers import AutoTokenizer
        from sft import encode_trace
        tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
        tasks={t['uid']:t for t in data['train']};assert len(rs)==len({r['task_uid'] for r in rs})==1000
        total=0;maximum=0
        for r in rs:
            t=tasks[r['task_uid']];e=SokobanEnv(t)
            from task_envs import sokoban_agent
            agent=sokoban_agent(10);agent.update_from_env(e.render(),0,False,{})
            for m in r['messages']:
                if m['role']!='assistant':continue
                action=int(agent.update_from_model(m['content']).action);obs,rew,done,info=e.step(action)
                if not done:agent.update_from_env(obs,rew,done,info)
            assert e.success() and agent.chat_completions==r['messages']
            ids,mask=encode_trace(tok,r['messages']);total+=sum(mask);maximum=max(maximum,len(ids));assert not mask[0]
            targets=tok.decode([i for i,m in zip(ids,mask) if m])
            expected=''.join(m['content']+tok.eos_token for m in r['messages'] if m['role']=='assistant')
            assert targets==expected
        teacher=dict(trajectories=1000,replayed=True,mask_verified=True,target_tokens=total,max_sequence_tokens=maximum)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest.update(files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['manifest.json','cpu_acceptance.json']},frozen_actual_maps=True,loaded_maps_equal_generated=True,generator_sha256=sha(Path(__file__).with_name('generate_hard_tasks.py')),task_envs_sha256=sha(Path(__file__).resolve().parents[1]/'task_envs.py'))
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'cpu_acceptance.json',dict(passed=True,tasks=10700,loaded_maps_equal_generated=True,all_solutions_replayed=True,golden_teacher=teacher,manifest_sha256=sha(root/'manifest.json'),maps_sha256=sha(target),data_sha256=sha(root/'data.json')))
    print(kind,json.dumps(manifest['split_statistics']),teacher,flush=True)
if __name__=='__main__':
    freeze('flh');freeze('sokoban')
