"""Predeclared one-round selectors; 'uct' is an adaptive mixture, not classical UCT."""
import hashlib,json,random
SELECTORS=('uniform','S','H_act','H_cons','uct')

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(list(parts),sort_keys=True).encode()).digest()[:8],'big')%(2**31-1)

def deduplicate(nodes):
    unique={}
    for n in sorted(nodes,key=lambda n:(n['j'],n['turn'])):
        key=tuple(n['branch'].ids)
        if key in unique:
            old=unique[key]
            assert old['state']==n['state']
            old['aliases'].append([n['j'],n['turn']])
        else:
            n['aliases']=[[n['j'],n['turn']]];unique[key]=n
    return list(unique.values())

def select(nodes,selector,forks,p_hat,base_seed,uid,step):
    assert selector in SELECTORS and forks in (1,2) and 0<=p_hat<=1
    ordered=sorted(nodes,key=lambda n:(n['j'],n['turn']))
    random.Random(stable_seed(base_seed,uid,step)).shuffle(ordered)
    def score(n):
        if selector=='uniform':return 0.
        if selector=='uct':return (1-p_hat)*n['S']+p_hat*n['H_cons']
        return n[selector]
    return sorted(ordered,key=lambda n:-score(n))[:forks]
