"""Recompute paired task-level summaries from committed evidence, without a GPU."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['main100','supplemental500'],default='main100')
p.add_argument('--bootstrap',type=int,default=50000);a=p.parse_args()
if a.bootstrap<1:raise SystemExit('--bootstrap must be positive')
root=Path(__file__).resolve().parents[2]/'results'/a.dataset
rows={}
for method in ['grpo','patr','tree_random']:
 path=root/'analysis'/f'{method}_per_task.json' if a.dataset=='main100' else root/method/'per_task.json'
 rows[method]=json.loads(path.read_text())
uids=[x['uid'] for x in rows['grpo']]
assert len(set(uids))==len(uids)
arrays={}
for method,data in rows.items():
 by_uid={x['uid']:x for x in data};assert set(by_uid)==set(uids)
 arrays[method]=np.array([[by_uid[u]['success_pct'],by_uid[u]['pass4_pct']] for u in uids])
 print(method,dict(zip(['Success_pct','Pass4_pct'],arrays[method].mean(0))))
for first,second in [('patr','grpo'),('patr','tree_random'),('tree_random','grpo')]:
 diff=arrays[first]-arrays[second];rng=np.random.default_rng(42);samples=[]
 for start in range(0,a.bootstrap,1000):
  idx=rng.integers(0,len(uids),(min(1000,a.bootstrap-start),len(uids)))
  samples.append(diff[idx].mean(1))
 ci=np.percentile(np.concatenate(samples),[2.5,97.5],axis=0)
 print(first+' minus '+second,{'difference_pp':diff.mean(0).tolist(),'CI95_lower':ci[0].tolist(),'CI95_upper':ci[1].tolist()})
print('Task-level paired percentile bootstrap; seed=42. Does not measure training-seed uncertainty. Dataset:',a.dataset)
