"""Frozen supplemental sets: reporting only, never parameter selection."""
import json,hashlib
from pathlib import Path
import numpy as np

def summary(rows):
 groups={}
 for row in rows:groups.setdefault(row['task']['uid'],[]).append(row)
 assert all(len(v)==4 for v in groups.values())
 return dict(n_tasks=len(groups),n_trajectories=len(rows),success_rate=sum(r['reward']>0 for r in rows)/len(rows),pass_at_4=sum(any(r['reward']>0 for r in rs) for rs in groups.values())/len(groups))

def evaluate_extra(engine,tok,cfg,out,evaluate):
 import patr_core as core
 root=Path(cfg.get('eval_sets_dir',str(Path(__file__).resolve().parents[1]/'eval_sets')));out=Path(out)
 raw=(root/'supplemental500/tasks.json').read_bytes();protocol=json.loads((root/'supplemental500/protocol.json').read_text())
 assert hashlib.sha256(raw).hexdigest()==protocol['task_file_sha256']
 tasks=json.loads(raw);rows=[];supp=out/'supplemental500';supp.mkdir(exist_ok=True)
 for i in range(5):
  dest=supp/f'batch{i+1}.json'
  evaluate(engine,tok,cfg,tasks[i*100:(i+1)*100],dest,protocol['evaluation_seed'],'supplemental500')
  rows.extend(json.loads(l) for l in dest.with_suffix('.jsonl').read_text().splitlines())
 core.atomic_json(supp/'summary.json',summary(rows))
 if cfg.get('environment') in ['flh','sokoban']:return
 filtered=json.loads((root/'filtered409/tasks.json').read_text());uids={r['uid'] for r in filtered};assert len(uids)==409
 core.atomic_json(out/'filtered409.json',dict(summary([r for r in rows if r['task']['uid'] in uids]),source='fixed subset of supplemental500; same sampled trajectories'))
 tasks=json.loads((root/'map8/tasks.json').read_text());protocol=json.loads((root/'map8/protocol.json').read_text())
 original=core.make_env
 def env8(task,cfg):
  e=core.FrozenLakeEnv(size=task['size'],p=task['p'],seed=task['seed'],is_slippery=cfg['is_slippery'],max_steps=8);e.reset();return e
 expected={m['uid']:m['map8'] for m in json.loads((root/'map8/source_maps.json').read_text())['maps']}
 for t in tasks:
  e=env8(t,cfg);assert [''.join(c.decode() for c in row) for row in e.desc]==expected[t['uid']]
 core.make_env=env8
 try:evaluate(engine,tok,cfg,tasks,out/'map8.json',protocol['evaluation_seed'],'map8')
 finally:core.make_env=original
