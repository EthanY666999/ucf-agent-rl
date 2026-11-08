#!/usr/bin/env python3
"""Independent replay of an already written Golden Trace file (CPU only)."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from alfworld_adapter import ALFWorldEpisode,parse_action

def replay(job):
    root,row=job
    game=Path(root)/row['game']
    assert hashlib.sha256(game.read_bytes()).hexdigest()==row['game_sha256']
    assert row['split']=='train'
    env=ALFWorldEpisode(game,row['seed'])
    try:
        assert env.observation==row['initial_observation']
        for step in row['steps']:
            assert env.observation==step['observation']
            assert env.admissible==step['admissible_actions']
            assert env.prompt()==step['prompt']
            assert parse_action(step['response'])==step['action']
            assert step['action'] in env.admissible
            obs,reward,done=env.step(step['action'])
            assert (obs,reward,done)==(step['next_observation'],step['reward'],step['done'])
        assert env.won
        return row['id']
    finally:env.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--games-root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('workers must be 1..4')
    with gzip.open(a.input,'rt') as f:rows=[json.loads(x) for x in f]
    manifest=json.loads(a.manifest.read_text());train=set(manifest['splits']['train'])
    assert len({x['id'] for x in rows})==len(rows)
    assert len({x['game'] for x in rows})==len(rows)
    assert all(x['game'] in train and x['split']=='train' for x in rows)
    if a.limit:rows=rows[:a.limit]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as executor:
        verified=list(executor.map(replay,[(str(a.games_root),row) for row in rows]))
    report=dict(status='passed',independently_replayed=len(verified),full_file=a.limit==0,
                gpu_used=False,input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__).resolve().parents[1]/'code/alfworld_adapter').glob('*.py')})
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
