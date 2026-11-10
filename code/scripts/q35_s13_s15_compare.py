"""Read-only validation comparison; no model loading, sampling or test selection."""
import json, hashlib, datetime
from pathlib import Path
RUNS=Path('/data/runs')
PATHS={'S13':RUNS/'hard_queue/S13_grpo_seed42','S15':RUNS/'hard_queue/S15_ucf_seed42'}
def summarize(records, tasks):
    grouped={t['uid']:[] for t in tasks}
    expected={t['uid']:t for t in tasks}
    for r in records:
        uid=r['task']['uid'];assert r['task']==expected[uid], 'Frozen validation task differs'
        assert r['reward'] in [0,1] and (r['reward']==1)==(r['terminal']=='success')
        grouped[uid].append(r)
    assert len(grouped)==100 and len(records)==400
    assert all(len(rs)==4 and {str(r['lineage']) for r in rs}=={'0','1','2','3'} for rs in grouped.values())
    wins=[sum(r['reward'] for r in rs) for rs in grouped.values()]
    return dict(success_rate_pct=sum(wins)/4,pass_at_4_pct=sum(w>0 for w in wins),mixed_tasks=sum(0<w<4 for w in wins),all_failure_tasks=sum(w==0 for w in wins),all_success_tasks=sum(w==4 for w in wins),mean_response_tokens=sum(r['response_tokens'] for r in records)/400,n_tasks=100,n_trajectories=400)
def build(paths=PATHS,dataset=RUNS/'datasets/sokoban/data.json'):
    tasks=json.loads(dataset.read_text())['validation'];points={}
    for ident,path in paths.items():
        points[ident]={}
        for step in range(0,401,25):
            summary=path/f'validation_{step:04d}.json';raw=summary.with_suffix('.jsonl')
            if not summary.exists() or not raw.exists():continue
            metric=json.loads(summary.read_text());value=summarize([json.loads(l) for l in raw.read_text().splitlines()],tasks)
            assert abs(metric['success_rate_pct']-value['success_rate_pct'])<1e-9 and abs(metric['pass_at_4_pct']-value['pass_at_4_pct'])<1e-9
            assert abs(metric['response_length']-value['mean_response_tokens'])<1e-9
            value['raw_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest();points[ident][step]=value
    steps=sorted(set(points['S13'])|set(points['S15']))
    return dict(checked_at=datetime.datetime.now().astimezone().isoformat(),split='fixed_validation',implementations={'S13':'noSFT GRPO4 first-complete-block','S15':'noSFT UCF-v2.2-ff causal-complement first-complete-block'},comparison_caveat='S13/S15 share new noSFT protocol; F02 is historical and not paired. Versus S05, initialization, protocol and implementation differ; versus future corrected SFT UCF, initialization and protocol differ',selection_use=False,group_definition='four validation lineages per frozen task; training leaf counts are separate',reply_length_unit='mean generated response tokens per trajectory, summed across turns',rows=[dict(step=s,S13=points['S13'].get(s),S15=points['S15'].get(s),paired=s in points['S13'] and s in points['S15']) for s in steps],not_started=[k for k,v in points.items() if not v])
def write_report():
    report=build();out=RUNS/'reports_q35/S13_S15_validation_comparison.json';out.parent.mkdir(exist_ok=True)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(out)
    return report
if __name__=='__main__':print(json.dumps(write_report(),ensure_ascii=False))
