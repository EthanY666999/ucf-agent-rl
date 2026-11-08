"""Reasoning-prefix + first-fence HT simulation, with explicit target/scope.

The supplied htsim_original.py and its regression remain unchanged. This adds
prefix-conditioned cases, including the residual p[0] which that four-action
simulation cannot represent. F=2 is divided out on BOTH estimator and target.
"""
import json
import numpy as np

def run(residual=0., N=200000, seed=20261109):
    rng=np.random.default_rng(seed)
    # Two sampled reasoning prefixes, each followed by a stochastic fence.
    prefix_mass=np.array([.4,.6]); fence_prob=np.array([.65,.85])
    valid=np.array([[.08,.07,.1,.15,.2,.1,.12,.18],[.15,.05,.1,.2,.05,.1,.25,.1]])
    probs=np.column_stack([np.full(2,residual),valid*(1-residual)])
    classes=np.array([0,0,0,1,1,2,2,3,3])
    rewards=np.array([[.15,.1,.1,.35,.35,.9,.9,.55,.55],[.05,.2,.2,.1,.1,.8,.8,.4,.4]])
    exact=np.zeros((2,9)); estimates=np.zeros((N,2,9)); copies=0
    for h in range(2):
        p=probs[h];q=rewards[h];V=p@q
        exact[h]=prefix_mass[h]*fence_prob[h]*p*(q-V)
    def draw(p):return np.minimum((rng.random(len(p))[:,None]>p.cumsum(1)).sum(1),8)
    # Four original trajectories, independent leave-one-out rewards/baselines.
    hs=rng.choice(2,(N,4),p=prefix_mass)
    aa=draw(probs[hs.ravel()]).reshape(N,4)
    Rs=(rng.random((N,4))<rewards[hs,aa]).astype(float)
    fences=rng.random((N,4))<fence_prob[hs]
    for j in range(4):
        h=hs[:,j];a=aa[:,j];base=(Rs.sum(1)-Rs[:,j])/3
        rho=np.clip(.6*((1-base)*np.array([.3,.7])[h]+base*.5),.03,1)
        pick=fences[:,j] & (rng.random(N)<rho)
        ids=np.flatnonzero(pick);hh=h[ids];a=a[ids];p=probs[hh]
        same=classes[a,None]==classes[None,:]
        po=(p*same).sum(1)
        allowed=~same;allowed[:,0]=False  # Authorized A_cf contains actions 1..4 only.
        other=p*allowed;pc=other.sum(1);assert (pc>0).all()
        x=draw(other/pc[:,None]);rc=(rng.random(len(ids))<rewards[hh,x]).astype(float)
        for token,value in [(a,po*(Rs[ids,j]-base[ids])/(4*rho[ids])),(x,pc*(rc-base[ids])/(4*rho[ids]))]:
            estimates[ids,hh]-=value[:,None]*p
            estimates[ids,hh,token]+=value
        copies+=len(ids)
    mean=estimates.mean(0);se=estimates.std(0,ddof=1)/np.sqrt(N);keep=se>0
    expected_estimator=exact.copy()
    mean_baseline=sum(prefix_mass[h]*(probs[h]@rewards[h]) for h in range(2))
    for h in range(2):
        p=probs[h];score0=-p.copy();score0[0]+=1
        stay_mass=p[classes==classes[0]].sum()
        expected_estimator[h]-=prefix_mass[h]*fence_prob[h]*p[0]*(1-stay_mass)*(rewards[h,0]-mean_baseline)*score0
    z=float((np.abs(mean-exact)[keep]/se[keep]).max())
    return dict(N=N,seed=seed,residual_mass=residual,max_abs_z=z,passed=z<4,
      max_z_against_analytic_estimator=float((np.abs(mean-expected_estimator)[keep]/se[keep]).max()),
      max_analytic_bias=float(np.abs(expected_estimator-exact).max()),
      mean_forks=copies/N,Part_B_reasoning_and_fence_gradient=0,
      target='exact conditional action-suffix gradient weighted by observed prefix/fence occupancy; F=2 common scale removed',
      prefix_probability_not_multiplied_again=True)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=[run(0.),run(.08)]
    report=dict(cases=cases,passed=all(x['passed'] for x in cases))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
