"""CPU-only LOO/full-batch c ratios from immutable S05 selection records."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ucf'))
from ucf_select import calibrate,RHO_MIN

def vector_loo(groups):
 keys=sorted(groups);w=np.array([w for k in keys for w in groups[k]],dtype=float);owners=np.array([k for k in keys for _ in groups[k]])
 mask=owners[None,:]!=np.array(keys)[:,None];n=len(keys)-1
 assert n>0
 reachable=(mask*np.where(w>0,1.,RHO_MIN)).sum(1)/n
 assert np.all(mask.sum(1)*RHO_MIN/n<=2.)
 def avg(c):return (np.clip(c[:,None]*w,RHO_MIN,1.)*mask).sum(1)/n
 lo=np.zeros(len(keys));hi=np.ones(len(keys));limited=reachable<=2.
 for i in np.flatnonzero(limited):
  pos=w[mask[i]&(w>0)];hi[i]=np.nextafter(np.max(1/pos),np.inf) if len(pos) else 0.
 while np.any((avg(hi)<2.)&~limited):hi[(avg(hi)<2.)&~limited]*=2
 for _ in range(80):
  mid=(lo+hi)/2;low=(avg(mid)<2.)&~limited;lo[low]=mid[low];high=~low&~limited;hi[high]=mid[high]
 return dict(zip(keys,np.where(limited,hi,(lo+hi)/2).tolist()))

def run(path,output):
 status=json.loads((path/'status.json').read_text());bound=status['step'];records=[];digests={};ratios=[]
 for file in sorted((path/'ucf_logs').glob('select_*.jsonl')):
  step=int(file.stem.split('_')[-1])
  if step>bound:continue
  rows=[json.loads(l) for l in file.read_text().splitlines()];groups={k:[] for k in range(64)}
  for r in rows:
   if not r['parse_mismatch']:groups[r['task_id']].append(r['w'])
  c0=rows[0]['c_step'];assert all(r['c_step']==c0 for r in rows)
  ck=vector_loo(groups)
  if step in [1,bound]:
   for k in [0,63]:assert abs(ck[k]-calibrate([ws for j,ws in groups.items() if j!=k]))<1e-12
  assert c0>0
  r=[ck[k]/c0 for k in sorted(ck)];ratios+=r
  records.append(dict(step=step,full_batch_c=c0,c_by_question=ck,ratios=r))
  digests[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
 assert len(records)==bound
 a=np.array(ratios);qs=[0,.01,.05,.25,.5,.75,.95,.99,1]
 result=dict(passed=True,experiment='S05 historical v2',through_step=bound,question_steps=len(a),quantiles=dict(zip(['min','p01','p05','p25','median','p75','p95','p99','max'],np.quantile(a,qs).tolist())),mean=float(a.mean()),std=float(a.std()),fraction_below1=float((a<1).mean()),fraction_above1=float((a>1).mean()),records=records,selection_sha256=digests,definition='Counterfactual recalibration on stored eligible weights only; does not change S05 or its training decisions',CPU_only=True)
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','selection_sha256']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run,a.output)
