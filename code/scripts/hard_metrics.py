"""Reporting-only protocol additions. Never used for selection or stopping."""
import json,gzip
from pathlib import Path
from collections import Counter

def group_counts(branches,n):
    groups={i:[] for i in range(n)}
    for b in branches:groups[b.task_id].append(b)
    # Flat collectors always provide the initial four completed trajectories.
    assert all(len(bs)==4 for bs in groups.values())
    counts=Counter('all_failure' if sum(b.reward>0 for b in bs)==0 else 'all_success' if sum(b.reward>0 for b in bs)==4 else 'mixed' for bs in groups.values())
    return dict(initial_group_counts={k:counts[k] for k in ['all_success','all_failure','mixed']},initial_group_fractions={k:counts[k]/n for k in ['all_success','all_failure','mixed']})

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault(r['task']['uid'],[]).append(r)
    assert all(len(rs)==4 and len({r['lineage'] for r in rs})==4 for rs in groups.values())
    return dict(n_tasks=len(groups),success_rate_pct=100*sum(r['reward'] for r in rows)/len(rows),pass_at_4_pct=100*sum(any(r['reward'] for r in rs) for rs in groups.values())/len(groups)) if groups else dict(n_tasks=0,success_rate_pct=None,pass_at_4_pct=None)

def strata(rows,cfg):
    from patr_core import make_env
    from task_envs import lake_metadata
    metadata={}
    for row in rows:
        task=row['task'];uid=task['uid']
        if uid in metadata:continue
        if 'shortest_path_length' in task:metadata[uid]=task
        else:metadata[uid]=lake_metadata(make_env(task,cfg))
    return dict(overall=summarize(rows),by_shortest_path={str(k):summarize([r for r in rows if metadata[r['task']['uid']]['shortest_path_length']==k]) for k in sorted({m['shortest_path_length'] for m in metadata.values()})},requires_detour=summarize([r for r in rows if metadata[r['task']['uid']]['requires_detour']]))

def completed_metrics(out,cfg):
    from patr_core import atomic_json
    out=Path(out);points={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in out.glob('validation_*.json')}
    complete=set(points)==set(range(0,401,25))
    result=dict(validation_point_count=len(points),validation_mean_17_points=dict(success_rate_pct=sum(p['success_rate_pct'] for p in points.values())/17,pass_at_4_pct=sum(p['pass_at_4_pct'] for p in points.values())/17) if complete else None,validation_mean_definition='arithmetic mean at steps 0,25,...,400; not trapezoidal integral',final_strata={})
    files={'main100':[out/'final_test.jsonl'],'supplemental500':sorted((out/'supplemental500').glob('batch*.jsonl'))}
    if (out/'map8.jsonl').exists():files['map8']=[out/'map8.jsonl']
    for label,paths in files.items():
        if paths and all(p.exists() for p in paths):
            rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
            result['final_strata'][label]=strata(rows,dict(cfg,max_turns=8) if label=='map8' else cfg)
            if label=='supplemental500' and cfg.get('environment','frozenlake')=='frozenlake':
                subset=json.loads((Path(cfg['eval_sets_dir'])/'filtered409/tasks.json').read_text());uids={t['uid'] for t in subset}
                result['final_strata']['filtered409']=strata([r for r in rows if r['task']['uid'] in uids],cfg)
    atomic_json(out/'additional_metrics.json',result)
    return result

def backfill_flat(out):
    """Sidecar for live F02: reads retained rollouts, never writes train_metrics."""
    from types import SimpleNamespace
    from patr_core import atomic_json
    out=Path(out);records=[]
    for path in sorted(out.glob('rollouts_*.jsonl.gz')):
        with gzip.open(path,'rt') as f:rows=[json.loads(s) for s in f]
        uids=sorted({r['task']['uid'] for r in rows});idx={u:i for i,u in enumerate(uids)}
        bs=[SimpleNamespace(task_id=idx[r['task']['uid']],reward=r['reward']) for r in rows]
        records.append(dict(step=int(path.name.split('_')[1].split('.')[0]),**group_counts(bs,len(uids))))
    atomic_json(out/'initial_group_counts_sidecar.json',dict(source='retained original rollouts; training process unchanged',steps=records))
    return records


def leaf_group_counts(branches,n,initial_not_applicable=True):
    groups={i:[b for b in branches if b.task_id==i] for i in range(n)}
    assert n>0 and all(groups.values()), 'Each sampled task must have at least one real leaf'
    counts=Counter('all_success' if all(b.reward>0 for b in bs) else 'all_failure' if not any(b.reward>0 for b in bs) else 'mixed' for bs in groups.values())
    result=dict(leaf_group_counts={k:counts[k] for k in ['all_success','all_failure','mixed']},leaf_group_fractions={k:counts[k]/n for k in ['all_success','all_failure','mixed']},leaf_group_source='all actual completed leaves; no pseudo rows or importance weighting',leaf_group_total=len(branches))
    if initial_not_applicable:result.update(initial_group_counts=None,initial_group_source='not defined: initial roots fork before termination; no diagnostic rollouts added')
    return result
