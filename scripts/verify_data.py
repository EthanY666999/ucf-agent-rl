"""Verify frozen splits and replay every published Sokoban/FLH demonstration."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from patr_core import make_env
from task_envs import sokoban_agent
from rllm.agents.frozenlake_agent import FrozenLakeAgent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify(env_name):
    root = ROOT/'data'/env_name
    data = json.loads((root/'data.json').read_text())
    task_by_uid = {t['uid']:t for t in data['train']}
    split_sets = {s:{t['uid'] for t in ts} for s,ts in data.items()}
    for s, uids in split_sets.items():
        assert len(uids) == len(data[s]), f'Duplicate UIDs in {s}'
        for other in split_sets:
            if other != s:
                assert not uids & split_sets[other], f'Split overlap: {s}/{other}'
    cfg = dict(max_turns=10 if env_name=='sokoban' else 20, is_slippery=False,
               frozen_maps_path=str(root/'maps.json'), frozen_maps_sha256=sha(root/'maps.json'))
    source = root/'sft_messages.jsonl' if env_name=='sokoban' else root/'golden_trace/sft_messages.jsonl.gz'
    seen=set(); action_count=0
    for row in rows(source):
        uid=row['task_uid']; assert uid not in seen and uid in task_by_uid
        seen.add(uid); env=make_env(task_by_uid[uid],cfg)
        agent=sokoban_agent(cfg['max_turns']) if env_name=='sokoban' else FrozenLakeAgent(max_steps=cfg['max_turns'])
        observation=env.render() if env_name=='sokoban' else env.reset()[0]
        agent.update_from_env(observation,0,False,{})
        assert agent.chat_completions==row['messages'][:2], f'Initial teacher context differs: {uid}'
        actions=[int(agent._parse_model_response(m['content'])[1]) for m in row['messages'] if m['role']=='assistant']
        assert 0<len(actions)<=cfg['max_turns'] and all(1<=a<=4 for a in actions)
        for i,a in enumerate(actions):
            message=row['messages'][2+2*i]
            assert int(agent.update_from_model(message['content']).action)==a
            observation, reward, done, info=env.step(a)
            if done: assert i==len(actions)-1, f'Trailing actions after termination: {uid}'
            else:
                agent.update_from_env(observation,reward,done,info)
                assert agent.chat_completions[-1]==row['messages'][3+2*i], f'Teacher observation differs: {uid}'
        assert env.success() and reward==1, f'Unsuccessful teacher replay: {uid}'
        assert agent.chat_completions==row['messages'], f'Teacher transcript differs: {uid}'
        action_count+=len(actions)
    assert len(seen)==1000
    return dict(environment=env_name, independently_replayed=1000, successful=1000,
                train_only=True, heldout_uid_overlap=0, action_count=action_count,all_observations_verified=True,
                source_sha256=sha(source), GPU_used=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    result=[verify(env) for env in ['sokoban','flh']]
    text=json.dumps(result,indent=2)+'\n';print(text)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)


if __name__=='__main__':main()
