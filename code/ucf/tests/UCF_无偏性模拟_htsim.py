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
    exact=np.zeros((2,9)); estimates=np.zeros((N,2,9)); copies=0; rejection_batches=0; exhaustion=0
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
        allowed=~same  # Full complement includes residual token 0 whenever its class differs.
        other=p*allowed;pc=other.sum(1);assert (pc>0).all()
        x=draw(other/pc[:,None]);failed=np.zeros(len(ids),dtype=bool)
        # STAY uses finite allowed tokens; non-STAY uses the first accepted
        # of exactly256 unrestricted proposals from the same prefix.
        for prefix in range(2):
            sel=np.flatnonzero((hh==prefix)&(classes[a]!=0));rejection_batches+=len(sel)
            for offset in range(0,len(sel),2048):
                ix=sel[offset:offset+2048]
                proposals=rng.choice(9,(len(ix),256),p=probs[prefix])
                accept=classes[proposals]!=classes[a[ix],None]
                ok=accept.any(1);chosen=accept.argmax(1)
                x[ix[ok]]=proposals[np.arange(len(ix))[ok],chosen[ok]]
                failed[ix[~ok]]=True
        exhaustion+=int(failed.sum())
        rc=(rng.random(len(ids))<rewards[hh,x]).astype(float)
        po=po.copy();pc=pc.copy();po[failed]=1.;pc[failed]=0.
        for token,value in [(a,po*(Rs[ids,j]-base[ids])/(4*rho[ids])),(x,pc*(rc-base[ids])/(4*rho[ids]))]:
            estimates[ids,hh]-=value[:,None]*p
            estimates[ids,hh,token]+=value
        copies+=len(ids)
    mean=estimates.mean(0);se=estimates.std(0,ddof=1)/np.sqrt(N);keep=se>0
    expected_estimator=exact.copy();analytic_bias=np.zeros_like(exact)
    mean_baseline=sum(prefix_mass[h]*(probs[h]@rewards[h]) for h in range(2))
    for h in range(2):
        p=probs[h];q=rewards[h];score0=-p.copy();score0[0]+=1
        stay_mass=p[classes==classes[0]].sum()
        # Finite rejection exhaustion falls back to the original row. Its
        # class-dependent probability is explicit, not claimed exactly zero.
        total_g=p*(q-mean_baseline)-(p@(q-mean_baseline))*p
        bias=np.zeros(9)
        for c in [1,2,3]:
            mass=p[classes==c].sum();weights=p*(classes==c)*(q-mean_baseline)
            conditional_g=(weights-weights.sum()*p)/mass
            bias+=mass*(mass**256)*(conditional_g-total_g)
        analytic_bias[h]=prefix_mass[h]*fence_prob[h]*bias
        expected_estimator[h]+=analytic_bias[h]
    z=float((np.abs(mean-exact)[keep]/se[keep]).max())
    return dict(N=N,seed=seed,residual_mass=residual,max_abs_z=z,passed=z<4,
      max_z_against_analytic_estimator=float((np.abs(mean-expected_estimator)[keep]/se[keep]).max()),
      max_analytic_bias=float(np.abs(analytic_bias).max()),
      rejection_batches=rejection_batches,rejection_exhausted=exhaustion,rejection_fallback_fraction=exhaustion/max(1,rejection_batches),finite_budget_fallback_not_an_exact_unbiasedness_claim=True,mean_forks=copies/N,Part_B_reasoning_and_fence_gradient=0,
      target='exact conditional action-suffix gradient weighted by observed prefix/fence occupancy; F=2 common scale removed',
      prefix_probability_not_multiplied_again=True)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=[run(0.),run(.08)]
    report=dict(cases=cases,passed=all(x['passed'] for x in cases))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
