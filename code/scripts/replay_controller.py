"""Replay saved randomness; compare v1 and per-batch v2 on identical candidates."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
from ucf_select import Controller,calibrate,RHO_MIN

def replay(root):
 paths=sorted(root.glob('select_*.jsonl'));assert len(paths)==400,(root,len(paths))
 init=json.loads((root/'controller_0000.json').read_text());ctrl=Controller(init['c'],init['history']);counts=[];checks=[];inputs={}
 for step,p in enumerate(paths,1):
  raw=p.read_bytes();inputs[p.name]=hashlib.sha256(raw).hexdigest();rows=[json.loads(l) for l in raw.splitlines()]
  assert {r['step'] for r in rows}=={step}
  groups=[[] for _ in range(64)]
  for row in rows:
   assert row['c_step']==ctrl.c,(step,row['c_step'],ctrl.c)
   rho=min(1.,max(RHO_MIN,ctrl.c*row['w']))
   assert rho==row['rho'] and (row['u']<rho)==row['selected'],(step,row)
   groups[row['task_id']].append(row['w'])
  ctrl.finish(sum(r['selected'] for r in rows)/64)
  recorded=json.loads((root/f'controller_{step:04d}.json').read_text())
  assert ctrl.c==recorded['c'] and ctrl.history==recorded['history'],step
  c=calibrate(groups,target=2.)
  counts.append(sum(r['u']<min(1.,max(RHO_MIN,c*r['w'])) for r in rows)/64)
  checks.append(dict(step=step,c_step=c,mean_forks=counts[-1]))
 q=np.quantile(counts,[.1,.5,.9]).tolist()
 return dict(passed=1.5<=q[0] and q[2]<=2.5,old_exact_steps=400,total_steps=400,new_p10_p50_p90=q,steps=checks,input_sha256=inputs,scope='Controller-only replay: original candidate set, scores and random uniforms unchanged; format exclusion is separately tested')
if __name__=='__main__':
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);results={}
 for name in ['golden-ucf-20261103','ucf-nosft-20261106']:
  root=Path('/data/runs/stage0/replay_inputs')/name/'code/runs/formal/ucf/code/ucf_logs'
  try: results[name]=replay(root)
  except Exception as e: results[name]=dict(passed=False,error=repr(e));raise
  finally:(out/'replay.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({k:{a:v for a,v in x.items() if a not in ['steps','input_sha256']} for k,x in results.items()},indent=2))
 assert all(v['passed'] for v in results.values())
