import numpy as np
rng=np.random.default_rng(0)
# 4x4 deterministic FrozenLake, horizon T
MAP=["S..H",".H..","...H","H..G"];H=W=4;T=8
cells=[(i,j) for i in range(H) for j in range(W)];idx={c:k for k,c in enumerate(cells)}
moves=[(0,-1),(1,0),(0,1),(-1,0)] # L D R U
def step(s,a):
    i,j=cells[s];di,dj=moves[a];ni,nj=i+di,j+dj
    if not(0<=ni<H and 0<=nj<W): ni,nj=i,j
    ns=idx[(ni,nj)];c=MAP[ni][nj]
    return ns,(1.0 if c=='G' else 0.0),(c in 'HG')
# BFS progress score
from collections import deque
g=idx[(3,3)];D={g:0};q=deque([g])
while q:
    u=q.popleft()
    for a in range(4):
        # reverse neighbors: any v with step(v,a)==u
        pass
dist=np.full(16,99)
for s in range(16):
    # BFS from s
    if MAP[cells[s][0]][cells[s][1]]=='H':continue
    D={s:0};q=deque([s])
    while q:
        u=q.popleft()
        if u==g:break
        for a in range(4):
            v,_,_=step(u,a);c=MAP[cells[v][0]][cells[v][1]]
            if v not in D and c!='H':D[v]=D[u]+1;q.append(v)
    dist[s]=D.get(g,99)
S=np.clip(1-dist/(H+W-2),0,1)
opt=np.zeros((16,4))
for s in range(16):
    best=None
    for a in range(4):
        ns,r,done=step(s,a)
        if MAP[cells[ns][0]][cells[ns][1]]=='H':continue
        if best is None or dist[ns]<dist[best[1]]:best=(a,ns)
    if best:opt[s,best[0]]=1
logits=2.2*opt+rng.normal(0,0.8,(16,4))
pi=np.exp(logits);pi/=pi.sum(1,keepdims=True)
start=idx[(0,0)]
# exact Q_t(s,a), V_t(s): t = turn index 0..T-1 (remaining T-t actions)
V=np.zeros((T+1,16));Q=np.zeros((T,16,4))
for t in range(T-1,-1,-1):
    for s in range(16):
        for a in range(4):
            ns,r,done=step(s,a);Q[t,s,a]=r+(0 if done else V[t+1,ns])
        V[t,s]=pi[s]@Q[t,s]
# visitation d_t(s)
d=np.zeros((T,16));d[0,start]=1
for t in range(T-1):
    for s in range(16):
        if d[t,s]==0:continue
        for a in range(4):
            ns,r,done=step(s,a)
            if not done:d[t+1,ns]+=d[t,s]*pi[s,a]
def glog(s,a):
    G=np.zeros((16,4));G[s]-=pi[s];G[s,a]+=1;return G
exact=np.zeros((16,4))
for t in range(T):
    for s in range(16):
        if d[t,s]==0:continue
        for a in range(4):exact+=d[t,s]*pi[s,a]*Q[t,s,a]*glog(s,a)
print('J=',V[0,start])
def rollout(s,t0):
    traj=[];R=0
    for t in range(t0,T):
        a=rng.choice(4,p=pi[s]);traj.append((t,s,a));ns,r,done=step(s,a);R+=r;s=ns
        if done:break
    return traj,R
def rollout_from(s,t0,first_a):
    ns,r,done=step(s,first_a);traj=[(t0,s,first_a)];R=r
    if not done and t0+1<T:
        tr,R2=rollout(ns,t0+1);traj+=tr;R+=R2
    return traj,R
def cls(s,a):
    ns,r,done=step(s,a);c=MAP[cells[ns][0]][cells[ns][1]]
    return 'FAIL' if c=='H' else ('GOAL' if c=='G' else ns)
def estimators(c_scale,rmin,use_phat):
    init=[rollout(start,0) for _ in range(4)]
    Rs=np.array([R for _,R in init])
    A=np.zeros((16,4))
    for jj,(tr,R) in enumerate(init):
        bl=np.delete(Rs,jj).mean()
        for t,s,a in tr:A+=(R-bl)*glog(s,a)
    A/=4
    B_ht=np.zeros((16,4));B_raw=np.zeros((16,4));B_cf=np.zeros((16,4));nsel=0
    for j,(tr,R) in enumerate(init):
        phat=np.delete(Rs,j).mean();bl=phat
        for k,(t,s,a) in enumerate(tr):
            w=S[s] if not use_phat else (1-phat)*S[s]+phat*0.5
            rho=float(np.clip(c_scale*w,rmin,1.0))
            if rng.random()<rho:
                nsel+=1
                tr2,R2=rollout(s,t)            # fresh on-policy continuation from node
                a2=tr2[0][2]
                B_ht+=(R2-bl)*glog(s,a2)/rho;B_raw+=(R2-bl)*glog(s,a2)
                # counterfactual stratified: original action a with return R (orig suffix), forced alt class
                co=cls(s,a);mask=np.array([cls(s,b)!=co for b in range(4)])
                pc=pi[s][~mask].sum()
                if mask.any():
                    mu=pi[s]*mask;mu/=mu.sum();acf=rng.choice(4,p=mu)
                    _,Rcf=rollout_from(s,t,acf)
                    B_cf+=(pc*(R-bl)*glog(s,a)+(1-pc)*(Rcf-bl)*glog(s,acf))/rho
                else:
                    B_cf+=(R-bl)*glog(s,a)/rho
    return A,B_ht/4,B_raw/4,B_cf/4,nsel
for use_phat,rmin in ((True,0.05),(True,0.15)):
    N=25000;acc={k:[] for k in 'A Bht Braw Bcf'.split()};ns=0
    for _ in range(N):
        A,b1,b2,b3,n=estimators(0.6,rmin,use_phat);ns+=n
        acc['A'].append(A);acc['Bht'].append(b1);acc['Braw'].append(b2);acc['Bcf'].append(b3)
    print('rho_min',rmin,'mean forks/task %.2f'%(ns/N))
    for k,v in acc.items():
        v=np.array(v);m=v.mean(0);se=v.std(0)/np.sqrt(N)
        rel=np.linalg.norm(m-exact)/np.linalg.norm(exact);cos=(m*exact).sum()/np.linalg.norm(m)/np.linalg.norm(exact)
        z=np.abs(m-exact)[se>0]/se[se>0]
        var=((v-m)**2).sum((1,2)).mean()
        print('  %-5s rel.err %.3f  cos %.3f  max|z| %.1f  total var %.4f'%(k,rel,cos,z.max(),var))
