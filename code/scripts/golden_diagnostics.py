"""Read-only diagnostics from preserved rollouts; does not alter RL rewards."""
import argparse
import collections
import gzip
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
results=[]
for f in sorted(a.run.glob('rollouts_*.jsonl.gz')):
    with gzip.open(f,'rt') as stream: rows=[json.loads(s) for s in stream]
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['task']['uid']].append(sum(t['reward'] for t in row['trace']))
    actions=[t['action'] for r in rows for t in r['trace']]
    first=[r['trace'][0]['action'] for r in rows if r['trace']]
    successes=[r for r in rows if sum(t['reward'] for t in r['trace'])>0]
    lengths=[len(r['trace']) for r in successes]
    results.append(dict(file=f.name,groups=len(groups),
       mixed_reward_group_fraction=sum(len(set(v))>1 for v in groups.values())/len(groups),
       first_turn_legal_fraction=sum(x in (1,2,3,4) for x in first)/len(first),
       invalid_or_missing_action_fraction=sum(x not in (1,2,3,4) for x in actions)/len(actions),
       successful_mean_turns=sum(lengths)/len(lengths) if lengths else None,
       note='Final retained branches; shared tree prefixes may repeat. Missing actions include token-limit responses.'))
(a.run/'golden_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
