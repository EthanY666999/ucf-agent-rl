"""Offline audit: did a token-stratum intervention change the Sokoban state?

Only selected training forks are represented. No policy or environment reward
is changed. Exact puzzle distances are diagnostic labels, never sampler inputs.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path


def audit(code, run, through):
    sys.path.insert(0, str(code))
    from sokoban_exact import distance_for_task
    config=json.loads((run/'config.json').read_bytes())
    base=json.loads((run/'initialization.json').read_bytes())['initial_step']
    assert config['environment']=='sokoban' and config['max_turns']==10 and config['repeat_limit']>config['max_turns']
    totals=Counter(); by_step=[]; pi=defaultdict(list)
    with (run/'branch_events.jsonl').open() as events:
        for step in range(base+1,through+1):
            line=json.loads(next(events));assert line['step']==step
            # Retain small state/label fields only. The recorded text is unnecessary.
            leaves={}; original_rewards=defaultdict(list); counts=Counter()
            path=run/'ucf_logs'/f'raw_leaves_{step:04}.jsonl.gz'
            with gzip.open(path,'rt') as f:
                for raw in f:
                    z=json.loads(raw);key=(z['task_id'],str(z['lineage']))
                    assert key not in leaves
                    # A truncated reply records the number of already executed
                    # moves, reusing the previous turn number. Candidate.turn
                    # indexes reply attempts, so preserve list order instead.
                    trace=[{k:t.get(k) for k in ('turn','action','anchor_before','anchor_after')} for t in z['trace']]
                    assert all(t['turn']==i+(t['action'] is not None) for i,t in enumerate(trace))
                    leaves[key]=(z['task'],z['reward'],z['terminal'],trace)
                    if z['kind']=='initial':original_rewards[z['task_id']].append(z['reward'])
                    assert z['terminal']!='pruned_repeat'
            assert len(original_rewards)==config['train_batch_size'] and all(len(rs)==4 for rs in original_rewards.values())
            for e in line['events']:
                assert e['kind']=='ucf_v3_fork' and e['step']==step and e['selected']
                ti,j,turn=e['task_id'],e['j'],e['turn']
                t,ro,_,o=leaves[ti,str(j)]; ot=o[turn-1]; assert ro==e['R_o']
                # For an unexecuted original turn, recover the pre-action state from the preceding turn.
                width=t['size']; initial=[t['player'][0]*width+t['player'][1], t['box'][0]*width+t['box'][1]]
                before=initial if turn==1 else o[turn-2]['anchor_after']
                assert before and (ot['anchor_before'] is None or ot['anchor_before']==before)
                remaining=config['max_turns']-turn+1
                distance=distance_for_task(t,divmod(before[0],width),divmod(before[1],width))
                feasible=distance is not None and distance<=remaining
                counts['selected']+=1;counts['feasible_pre_action']+=feasible
                counts['all_originals_failed']+=not any(original_rewards[ti])
                counts['all_failed_and_infeasible']+=not any(original_rewards[ti]) and not feasible
                if e['stratum']['cf_fallback']:
                    assert e['R_cf'] is None;counts['fallback']+=1;continue
                tc,rc,_,c=leaves[ti,f'{j}@{turn}.cf'];ct=c[turn-1]
                assert tc==t and rc==e['R_cf'] and (ct['anchor_before'] is None or ct['anchor_before']==before)
                assert feasible or (ro==0 and rc==0)
                counts['with_cf']+=1;counts['outcome_flip']+=ro!=rc
                rescue=not ro and bool(rc);counts['original_failure_cf_success']+=rescue
                p=e['stratum']['pi_cf'];assert 0<p<=1 and 0<e['rho']<=1
                pi['all'].append(p)
                if rescue: pi['original_failure_cf_success'].append(p)
                counts['zero_advantage_on_both_rows']+=ro==e['baseline'] and rc==e['baseline']
                counts['cf_token_partition_parser_disagreement']+=e['cf_parse_mismatch']
                if ot['action'] is None or ct['action'] is None:
                    counts['at_least_one_action_not_executed']+=1
                else:
                    assert ot['anchor_after'] is not None and ct['anchor_after'] is not None
                    counts['both_execute']+=1
                    same=ot['anchor_after']==ct['anchor_after']
                    counts['same_next_board']+=same
                    counts['different_next_board']+=not same
                    counts['flip_despite_same_next_board']+=same and ro!=rc
            assert counts['with_cf']==counts['both_execute']+counts['at_least_one_action_not_executed']
            assert counts['selected']==counts['with_cf']+counts['fallback']
            totals.update(counts);by_step.append(dict(step=step,counts=dict(counts)))
    return dict(through=through,run=str(run),counts=dict(totals),by_step=by_step,
                pi_cf={k:dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v)) for k,v in pi.items()},
                scope='Actual sampled forks, not all candidate nodes. A change in the first environment transition and an eventual reward flip are distinct. Stochastic tails can differ even after identical next boards. No final test or model weights were used.',
                sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),code/'sokoban_exact.py',run/'config.json']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--through',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=audit(a.code,a.run,a.through);tmp=a.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(a.out);print(json.dumps(d['counts']))
