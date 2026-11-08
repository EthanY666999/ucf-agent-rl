"""CPU-only UCF selection; probabilities fixed before any continuation."""
import json,math,os,random
from pathlib import Path
from cert_select import stable_seed
from cert_entropy import consequence_classes
RHO_MIN=.03
class Controller:
 def __init__(self,c0,history=()):
  assert math.isfinite(c0) and c0>0
  self.c=float(c0);self.history=list(history)
 def finish(self,mean_forks):
  assert math.isfinite(mean_forks) and mean_forks>=0
  self.history.append(float(mean_forks));m=sum(self.history[-20:])/len(self.history[-20:])
  self.c*=min(1.25,max(.8,2/m if m else 1.25))
  return self.c

def feature(node,first,variant='ucf'):
 others=[b for b in first if b.task_id==node['branch'].task_id and int(b.lineage)!=node['j']]
 assert len(others)==3
 ph=sum(b.terminal=='success' for b in others)/3
 baseline=sum(b.reward for b in others)/3
 w=1. if variant=='uniform' else (1-ph)*node['S']+ph*node['H_cons']
 assert math.isfinite(w) and 0<=w<=1
 return w,ph,baseline

def select_all(candidates,first,tasks,c_step,base_seed,step,path,variant='ucf'):
 rows=[];picked={ti:[] for ti in candidates}
 for ti,ns in candidates.items():
  for n in ns:
   w,ph,b=feature(n,first,variant);excluded=n.get('stratum',{}).get('parse_mismatch',False);rho=0. if excluded else min(1.,max(RHO_MIN,c_step*w))
   u=random.Random(stable_seed(base_seed,tasks[ti]['uid'],n['j'],n['turn'],step,'select')).random()
   n.update(w=w,rho=rho,u=u,selected=u<rho,p_hat_minus_j=ph,baseline=b,c_step=c_step)
   rows.append(dict(task_id=ti,uid=tasks[ti]['uid'],j=n['j'],turn=n['turn'],step=step,w=w,rho=rho,u=u,selected=u<rho,p_hat_minus_j=ph,baseline=b,c_step=c_step,parse_mismatch=excluded))
   if u<rho:picked[ti].append(n)
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
  f.flush();os.fsync(f.fileno())
 return picked,rows

def stratum(node,response_ids,mapping,fence):
 cls=consequence_classes(node['branch'].env)
 a=mapping.get(response_ids[1],0) if len(response_ids)>1 and response_ids[0]==fence else 0
 co=cls[a];p=node['pi'];allowed=[a for a in range(1,5) if cls[a]!=co]
 po=sum(p[a] for a in range(5) if cls[a]==co)
 pc=sum(p[a] for a in allowed);missing=p[0] if co!='STAY' else 0.
 assert abs(po+pc+missing-1)<1e-6
 return dict(first_action=a,original_class=co,allowed_actions=allowed,allowed_tokens=sorted(t for t,a in mapping.items() if a in allowed),pi_o=po,pi_cf=pc,unsampled_mass=missing,cf_fallback=not allowed)

def calibrate(task_weights,target=2.):
 def avg(c):return sum(sum(min(1,max(RHO_MIN,c*w)) for w in ws) for ws in task_weights)/len(task_weights)
 lo=0.;hi=1.
 assert avg(0)<=target,'rho_min alone exceeds target'
 achievable=sum(sum(1. if w>0 else RHO_MIN for w in ws) for ws in task_weights)/len(task_weights)
 # User-approved 2026-11-07: saturate eligible support when target is unreachable.
 # Zero-score candidates retain rho_min; format-excluded nodes are absent here.
 if achievable<=target:
  positive=[w for ws in task_weights for w in ws if w>0]
  return math.nextafter(max(1./w for w in positive),math.inf) if positive else 0.
 while avg(hi)<target:hi*=2
 for _ in range(80):
  mid=(lo+hi)/2
  if avg(mid)<target:lo=mid
  else:hi=mid
 return (lo+hi)/2
