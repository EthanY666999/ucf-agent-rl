"""Reporting-only paired task bootstrap; never feeds back into training."""
import argparse
import json
from collections import deque
from pathlib import Path
import numpy as np


def task_scores(rows):
    groups={}
    for row in rows:groups.setdefault(row['task']['uid'],[]).append(row)
    result={}
    for uid,rs in groups.items():
        assert len(rs)==4 and len({str(r['lineage']) for r in rs})==4,(uid,'expected four distinct trajectories')
        assert all(r['task']==rs[0]['task'] and r['reward'] in [0,1] for r in rs)
        result[uid]=dict(task=rs[0]['task'],success=sum(r['reward'] for r in rs)/4,pass4=float(any(r['reward'] for r in rs)))
    return result


def bootstrap(left,right,uids,replicates=10000):
    assert uids and len(set(uids))==len(uids)
    uids=sorted(uids)
    assert all(left[u]['task']==right[u]['task'] for u in uids)
    diffs=np.array([[left[u][k]-right[u][k] for k in ['success','pass4']] for u in uids])
    # Analysis-only RNG: does not touch model, rollout or evaluation seed state.
    rng=np.random.default_rng(20261107);samples=[]
    for start in range(0,replicates,250):
        index=rng.integers(0,len(uids),size=(min(250,replicates-start),len(uids)))
        samples.append(diffs[index].mean(axis=1))
    draws=np.concatenate(samples);ci=np.quantile(draws,[.025,.975],axis=0)
    return dict(n_tasks=len(uids),replicates=replicates,
                **{k:dict(difference_percentage_points=float(diffs[:,i].mean()*100),
                          percentile_95_ci_percentage_points=(ci[:,i]*100).tolist()) for i,k in enumerate(['success','pass4'])})


def shortest_path(desc):
    cells=[[x.decode() if isinstance(x,bytes) else str(x) for x in row] for row in desc]
    start=next((r,c) for r,row in enumerate(cells) for c,x in enumerate(row) if x=='S')
    queue=deque([(start,0)]);seen={start}
    while queue:
        (r,c),distance=queue.popleft()
        if cells[r][c]=='G':return distance
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<len(cells) and 0<=nc<len(cells[0]) and cells[nr][nc]!='H' and (nr,nc) not in seen:
                seen.add((nr,nc));queue.append(((nr,nc),distance+1))
    raise ValueError('No valid path in fixed evaluation map')


def read_rows(run):
    result=[]
    for i in range(1,6):
        result.extend(json.loads(line) for line in (run/f'supplemental500/batch{i}.jsonl').read_text().splitlines())
    return result


def report(left_run,right_run):
    root=Path(__file__).resolve().parents[1]
    cfgs=[json.loads((p/'config.json').read_text()) for p in [left_run,right_run]]
    for key in ['model','seed','max_response_tokens','eval_samples','eval_temperature','eval_top_p','eval_top_k']:
        assert cfgs[0][key]==cfgs[1][key],('Unmatched comparison',key)
    left,right=task_scores(read_rows(left_run)),task_scores(read_rows(right_run))
    setsroot=Path(cfgs[0].get('eval_sets_dir',root/'eval_sets'))
    assert cfgs[0].get('environment')==cfgs[1].get('environment')
    expected={r['uid'] for r in json.loads((setsroot/'supplemental500/tasks.json').read_text())}
    assert set(left)==set(right)==expected and len(expected)==500
    subsets=[('supplemental500',expected)]
    if (setsroot/'filtered409/tasks.json').exists():
        filtered={r['uid'] for r in json.loads((setsroot/'filtered409/tasks.json').read_text())};assert len(filtered)==409 and filtered<=expected
        subsets.append(('filtered409',filtered))
    import sys
    sys.path.insert(0,str(root))
    import patr_core
    from task_envs import lake_metadata
    metadata={u:(left[u]['task'] if 'shortest_path_length' in left[u]['task'] else lake_metadata(patr_core.make_env(left[u]['task'],cfgs[0]))) for u in sorted(expected)}
    lengths={u:metadata[u]['shortest_path_length'] for u in expected}
    result=dict(left=str(left_run),right=str(right_run),direction='left minus right',
                matched_seed=cfgs[0]['seed'],matched_initial_model=cfgs[0]['model'],
                reporting_only=True,bootstrap_rng_seed=20261107,task_shortest_paths=lengths,sets={})
    for name,uids in subsets:
        detours=[u for u in uids if metadata[u]['requires_detour']]
        result['sets'][name]=dict(overall=bootstrap(left,right,list(uids)),requires_detour=bootstrap(left,right,detours) if detours else None,
                by_exact_shortest_path={str(length):bootstrap(left,right,sorted(u for u in uids if lengths[u]==length)) for length in sorted({lengths[u] for u in uids})})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',required=True,type=Path);p.add_argument('--right',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=report(a.left,a.right);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
