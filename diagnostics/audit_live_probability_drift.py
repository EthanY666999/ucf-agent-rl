"""Read-only, complete-step audit of replay-probe vs actual sampled logprob."""
import argparse
import hashlib
import json
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
metrics_path=a.run/'train_metrics.jsonl'
metrics=[json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]
completed={x['step'] for x in metrics}
events=[]
with (a.run/'branch_events.jsonl').open() as f:
    for line in f:
        if not line.endswith('\n'):continue
        row=json.loads(line)
        if row['step'] in completed:
            events.extend(n for n in row['events'] if n['kind']=='ucf_v3_fork')
outliers=[]
for n in events:
    if abs(n['probe_original_logprob_difference'])<=.125:continue
    r={k:n[k] for k in ['step','task_id','j','turn','boundary','baseline','rho',
                         'R_o','R_cf','probe_original_logprob_difference']}
    s=n['stratum']
    r['original_advantage']=2*s['pi_o']*(n['R_o']-n['baseline'])
    r['counterfactual_advantage']=(None if n['R_cf'] is None else
                                  2*s['pi_cf']*(n['R_cf']-n['baseline']))
    r['both_pg_advantages_zero']=(r['original_advantage']==0 and
                                  r['counterfactual_advantage'] in (None,0))
    outliers.append(r)
steps=[]
for row in metrics:
    ds=row.get('probe_original_logprob_differences',[])
    steps.append(dict(step=row['step'],nodes=len(ds),mean_abs_logprob_delta=sum(map(abs,ds))/max(1,len(ds)),
                      max_abs_logprob_delta=max(map(abs,ds),default=0),
                      above_max_admission_limit=sum(abs(x)>.125 for x in ds),
                      training_success_rate=row['success_rate'],kl=row['kl'],grad_norm=row['grad_norm']))
result=dict(unix=time.time(),run=str(a.run),initial_gpu_thresholds=dict(mean_abs=.02,max_abs=.125),
            steps=steps,outliers=outliers,total_nodes=len(events),
            outliers_with_nonzero_pg=sum(not x['both_pg_advantages_zero'] for x in outliers),
            interpretation='Passing a small initial smoke did not bound all live replay-probe differences. '
              'Zero observed PG at an outlier is not proof of no distributional or KL effect. '
              'No threshold has been relaxed and no learning improvement is claimed.',
            source_metrics_bytes=metrics_path.stat().st_size)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed_steps=len(steps),total_nodes=len(events),outliers=len(outliers),
                      outliers_with_nonzero_pg=result['outliers_with_nonzero_pg'])))
