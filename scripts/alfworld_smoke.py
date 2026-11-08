#!/usr/bin/env python3
"""Real CPU environment, clone and hand-coded expert acceptance (no model)."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from alfworld_adapter import ALFWorldEpisode

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--games-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.manifest.read_text());results=[]
    for split in ['train','valid_seen','valid_unseen']:
        for game in data['splits'][split][:a.count]:
            env=ALFWorldEpisode(a.games_root/game,42)
            before=random.getstate();clone=env.fork()
            assert before==random.getstate(), 'Environment fork changed caller RNG'
            try:
                assert env.observation==clone.observation
                steps=0
                while not env.done:
                    action=env.expert_action()
                    assert action==clone.expert_action()
                    assert env.step(action)==clone.step(action)
                    steps+=1
                results.append(dict(game=game,split=split,steps=steps,won=env.won,
                                    fork_replay_passed=True,caller_rng_preserved=True))
            finally:
                env.close();clone.close()
    report=dict(status='passed',gpu_used=False,model_loaded=False,
                episodes=results,expert_successes=sum(x['won'] for x in results),
                note='Expert failures within 50 steps are retained; no success claim for every task.')
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
