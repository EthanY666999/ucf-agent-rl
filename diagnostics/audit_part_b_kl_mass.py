"""Read-only coefficient audit. These are not KL values or gradient norms."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

def response_tail(mask,boundary):
    assert mask[boundary]==1
    end=boundary
    while end<len(mask) and mask[end]:end+=1
    return end-boundary

def inspect(run):
    cfg=json.loads((run/'config.json').read_text());beta=cfg['ucf_beta'];n_tasks=cfg['train_batch_size']
    assert cfg['mini_batch_tasks']<=n_tasks and n_tasks%cfg['mini_batch_tasks']==0
    result=[]
    for line in (run/'branch_events.jsonl').read_text().splitlines():
        eventrow=json.loads(line);step=eventrow['step']
        p=run/'ucf_logs'/f'raw_leaves_{step:04d}.jsonl.gz'
        initial={};counterfactual={}
        with gzip.open(p,'rt') as f:
            for line in f:
                row=json.loads(line)
                target=initial if row['kind']=='initial' else counterfactual
                target[(row['task_id'],row['lineage'])]=dict(mask=row['mask'],reward=row['reward'])
        assert len(initial)==n_tasks*4
        mass=Counter();tokens=Counter();zero_nodes=0
        for e in eventrow['events']:
            assert e['kind']=='ucf_v3_fork'
            original=initial[(e['task_id'],str(e['j']))]
            original_tokens=sum(original['mask']);weight=beta/(e['rho']*n_tasks*4*original_tokens)
            s=e['stratum'];base=e['baseline'];fallback=s['cf_fallback']
            orig_adv=2*(1 if fallback else s['pi_o'])*(original['reward']-base)
            pairs=[('original',original,orig_adv)]
            if not fallback:
                child=counterfactual[(e['task_id'],f"{e['j']}@{e['turn']}.cf")]
                cf_adv=2*s['pi_cf']*(child['reward']-base)
                pairs.append(('counterfactual',child,cf_adv))
            zero_nodes+=all(adv==0 for _,_,adv in pairs)
            for kind,row,adv in pairs:
                length=response_tail(row['mask'],e['boundary'])
                tokens[kind]+=length;mass[kind]+=weight*length
                mass['total']+=weight*length
                if adv==0:mass['zero_pg_advantage']+=weight*length
        result.append(dict(step=step,original_trajectories=len(initial),fork_nodes=len(eventrow['events']),
            part_a_normalized_kl_coefficient_mass=1.,part_b_normalized_kl_coefficient_mass=dict(mass),
            combined_coefficient_mass=1.+mass['total'],part_b_mask_tokens=dict(tokens),
            zero_pg_nodes=zero_nodes,
            zero_pg_fraction_of_b_kl_mass=mass['zero_pg_advantage']/mass['total'] if mass['total'] else None))
    return dict(run=str(run),config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
        code_formula='loss = (PG + kl_coefficient * KL) * row_weight for both A and B; B row_weight=beta/(rho*n_tasks*4*original_response_tokens)',
        interpretation='Exact coefficient/token accounting on recorded samples, not actual KL magnitude or gradient direction. A zero PG advantage does not disable KL in the implemented loss. This alone cannot establish the cause of truncation.',
        steps=result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert response_tail([0,0,1,1,1,0,1],2)==3
    result=inspect(a.run);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([x for x in result['steps'] if x['step'] in (1,7,13,16)]))
