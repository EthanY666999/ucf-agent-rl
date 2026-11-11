"""Compare uniform and feasibility-gated selections on identical logged prefixes and draws."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def replay(code, run, start, end):
    sys.path[:0] = [str(code), str(code/'ucf')]
    from sokoban_exact import distance_for_task
    from v3_sampling import question_probabilities

    cfg = read(run/'config.json')
    assert cfg['environment'] == 'sokoban' and cfg['ucf_variant'] == 'uniform'
    assert cfg['initial_branches'] == 4 and cfg['max_turns'] == 10
    metrics = {r['step']:r for r in map(json.loads,(run/'train_metrics.jsonl').read_text().splitlines())}
    outputs = []
    sums = Counter()
    for step in range(start,end+1):
        logs = run/'ucf_logs'
        states, outcomes, rescued = {}, defaultdict(dict), set()
        with gzip.open(logs/f'raw_leaves_{step:04d}.jsonl.gz','rt') as file:
            for line in file:
                r = json.loads(line)
                ti = r['task_id']
                if r['kind'] == 'counterfactual':
                    if r['reward'] > 0:
                        rescued.add(ti)
                    continue
                assert r['kind']=='initial'
                j = int(r['lineage'])
                assert j not in outcomes[ti]
                outcomes[ti][j] = int(r['reward']>0)
                task = r['task']
                width = task['size']
                before = tuple(pos[0]*width+pos[1] for pos in (task['player'],task['box']))
                for turn,t in enumerate(r['trace'],start=1):
                    if t.get('anchor_before') is not None:
                        assert tuple(t['anchor_before']) == before
                    dist = distance_for_task(task,divmod(before[0],width),divmod(before[1],width))
                    states[ti,j,turn] = (dist, cfg['max_turns']-turn+1, task['uid'])
                    if t.get('action') is not None:
                        before = tuple(t['anchor_after'])
        assert len(outcomes)==cfg['train_batch_size'] and all(set(o)=={0,1,2,3} for o in outcomes.values())
        failed = {ti for ti,o in outcomes.items() if not any(o.values())}
        rows = [json.loads(line) for line in (logs/f'select_{step:04d}.jsonl').read_text().splitlines()]
        counts = dict.fromkeys(outcomes,0)
        eligible = counts.copy()
        marked = []
        seen = set()
        for r in rows:
            key = (r['task_id'],r['j'],r['turn'])
            assert key not in seen
            seen.add(key)
            dist,remaining,uid = states[key]
            assert uid==r['uid'] and r['step']==step
            can_finish = dist is not None and dist <= remaining
            counts[key[0]] += 1
            eligible[key[0]] += int(can_finish)
            marked.append((r, can_finish, dist is None))
        old = question_probabilities(counts,cfg['ucf_target_forks'])
        new = question_probabilities(eligible,cfg['ucf_target_forks'])
        ct = Counter(tasks=len(outcomes), candidates=len(rows), all_failed_groups=len(failed),
                     rescued_all_failed_groups=len(rescued & failed), zero_eligible_tasks=sum(n==0 for n in eligible.values()))
        for r,can_finish,structural in marked:
            ti = r['task_id']
            assert old[ti]==r['rho'] and (r['u']<old[ti])==r['selected']
            new_rho = new[ti] if can_finish else 0.
            selected = bool(r['selected'])
            gated = r['u'] < new_rho
            assert not gated or can_finish
            assert not (can_finish and selected) or gated
            ct['original_selected'] += selected
            ct['gated_selected_same_draw'] += gated
            ct['gated_newly_selected_unobserved'] += gated and not selected
            ct['original_unreachable_selected'] += selected and not can_finish
            ct['original_structural_deadlock_selected'] += selected and structural
            ct['original_failed_group_selected'] += selected and ti in failed
            ct['original_failed_group_unreachable_selected'] += selected and ti in failed and not can_finish
            ct['gated_unreachable_selected'] += gated and not can_finish
            ct['gated_expected'] += new_rho
            ct['uniform_expected'] += old[ti]
            if r['turn'] <= 3 and can_finish:
                ct['early_feasible_candidates'] += 1
                ct['uniform_early_feasible_expected'] += old[ti]
                ct['gated_early_feasible_expected'] += new_rho
        row = metrics[step]
        assert ct['original_selected']==row['branch_events']
        assert ct['all_failed_groups']==row['initial_group_counts']['all_failure']
        if failed:
            assert math.isclose(ct['rescued_all_failed_groups']/len(failed),row['rescue_rate'])
        sums.update(ct)
        outputs.append(dict(step=step,counts=dict(ct)))
    return dict(start=start,end=end,training_seed=cfg['seed'],source_run=str(run),counts=dict(sums),
                sources={f:hashlib.sha256((code/f).read_bytes()).hexdigest() for f in ('sokoban_exact.py','ucf/v3_sampling.py')},
                rule='Pre-action primitive distance <= remaining environment moves, then original leave-one-question-out uniform calibration on eligible counts. Use recorded selection draw.',
                interpretation='Frozen trajectory replay. A newly selected alternative has no observed counterfactual reward. Rescue counts and validation outcomes are observed only for the original policy.',
                by_step=outputs)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--steps',nargs=2,type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=replay(args.code,args.run,*args.steps)
    temp=args.output.with_suffix('.writing')
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(args.output)
    print(json.dumps(result['counts']))
