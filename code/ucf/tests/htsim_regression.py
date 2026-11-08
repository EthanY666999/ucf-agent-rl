"""Vectorized form of the supplied tabular simulation; fixed N and seed."""
import json,time
from pathlib import Path
import numpy as np
start_time=time.monotonic()
ns={};source=(Path(__file__).parent/'htsim_original.py').read_text();exec(source.split('for use_phat,rmin in')[0],ns)
pi,Q,exact=ns['pi'],ns['Q'],ns['exact'];N=25000;T=ns['T'];rng=np.random.default_rng(20261103)
nexts=np.array([[ns['step'](s,a)[0] for a in range(4)] for s in range(16)])
rews=np.array([[ns['step'](s,a)[1] for a in range(4)] for s in range(16)])
dones=np.array([[ns['step'](s,a)[2] for a in range(4)] for s in range(16)])
labels=[[ns['cls'](s,a) for a in range(4)] for s in range(16)]
eq=np.array([[[x==y for y in row] for x in row] for row in labels])
def sample(p):return np.minimum((rng.random(len(p))[:,None]>np.cumsum(p,axis=1)).sum(1),3)
def roll(states,t0,forced=None):
 s=states.copy();alive=np.ones(len(s),dtype=bool);R=np.zeros(len(s));tr=[]
 for t in range(t0,T):
  ids=np.where(alive)[0]
  if not len(ids):break
  a=forced[ids] if forced is not None and t==t0 else sample(pi[s[ids]])
  tr.append((t,ids.copy(),s[ids].copy(),a.copy()));R[ids]+=rews[s[ids],a];alive[ids]=~dones[s[ids],a];s[ids]=nexts[s[ids],a]
 return R,tr
initial=[roll(np.full(N,ns['start']),0) for _ in range(4)];Rs=np.stack([x[0] for x in initial],axis=1)
acc={k:np.zeros((N,16,4)) for k in ['A','Bht','Braw','Bcf']};forks=0
def add(name,ids,s,a,val):
 acc[name][ids,s,:]-=val[:,None]*pi[s]
 acc[name][ids,s,a]+=val
for j,(R,tr) in enumerate(initial):
 base=(Rs.sum(1)-R)/3
 for t,ids,s,a in tr:
  add('A',ids,s,a,(R[ids]-base[ids])/4)
  w=(1-base[ids])*ns['S'][s]+base[ids]*.5;rho=np.clip(.6*w,.03,1)
  pick=rng.random(len(ids))<rho;ids,s,a,rho=ids[pick],s[pick],a[pick],rho[pick];forks+=len(ids)
  if not len(ids):continue
  rn,tn=roll(s,t);an=tn[0][3]
  add('Braw',ids,s,an,(rn-base[ids])/4);add('Bht',ids,s,an,(rn-base[ids])/(4*rho))
  same=eq[s,a];po=(pi[s]*same).sum(1);other=pi[s]*(~same);pc=other.sum(1);valid=pc>0
  add('Bcf',ids,s,a,np.where(valid,po,1)*(R[ids]-base[ids])/(4*rho))
  if valid.any():
   ac=sample(other[valid]/pc[valid,None]);rc,_=roll(s[valid],t,forced=ac)
   add('Bcf',ids[valid],s[valid],ac,pc[valid]*(rc-base[ids[valid]])/(4*rho[valid]))
report=dict(seed=20261103,N=N,rho_min=.03,mean_forks=forks/N,estimators={})
for k,v in acc.items():
 mean=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(N);keep=se>0;z=float((np.abs(mean-exact)[keep]/se[keep]).max())
 report['estimators'][k]=dict(max_abs_z=z,relative_error=float(np.linalg.norm(mean-exact)/np.linalg.norm(exact)))
report['seconds']=time.monotonic()-start_time
report['passed']=report['estimators']['Bht']['max_abs_z']<4 and report['estimators']['Bcf']['max_abs_z']<4 and report['estimators']['Braw']['max_abs_z']>20 and report['seconds']<120
p=Path(__file__).resolve().parents[1]/'artifacts/htsim_result.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert report['passed'],report
