"""CPU-only completion reports. Training never consumes these outputs."""
import argparse
import hashlib
import json
import time
import subprocess
from pathlib import Path
import numpy as np
from paired_report import report

RUNS=Path('/data/runs');OUT=RUNS/'reports_q35'
PAIRS=[('H05',r) for r in ['H03','H04','H06','H07']]+[('H10',r) for r in ['H08','H09','H11','H12']]+[('S05',r) for r in ['S03','S04','S06','S07']]+[('S10',r) for r in ['S08','S09']]+[(r,'H05') for r in ['H13','H14','H15','H16']]+[('F06',r) for r in ['F05','F04','F07','F08']]+[('F10',r) for r in ['F09','F02']]+[('F11',r) for r in ['F12','F13','F14','F15']]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def diagnostics(records):
    # Retain the latest committed row for each step; raw attempt rows remain on disk.
    rs=list({r['step']:r for r in records}.values());rs.sort(key=lambda r:r['step'])
    norms=[r['ucf_preupdate_fullbatch_components']['B']['grad_norm'] for r in rs if 'ucf_preupdate_fullbatch_components' in r]
    return dict(steps=len(rs),curve_rows=rs,
                B_grad_norm=dict(n=len(norms),median=float(np.median(norms)),p99=float(np.quantile(norms,.99)),maximum=max(norms),steps_above3=sum(x>3 for x in norms)) if norms else None,
                steps_KL_above10=sum(r.get('kl',0)>10 for r in rs),
                plot_rendering='handled by separate plotting job; numeric source retained')

def available():
    result={}
    for folder in [RUNS/f'stage{stage}' for stage in range(1,6)]+[RUNS/'hard_queue']:
        for p in folder.glob('[FHS][0-9][0-9]_*'):
            if '_failed_' in p.name:continue
            receipt=p/'verification.json'
            if receipt.exists() and read(receipt).get('passed'):
                ident=p.name[:3];assert ident not in result,('Duplicate verified experiment',ident)
                result[ident]=p
    return result

def guarded(key,fingerprint,fn):
    path=OUT/'jobs'/(key+'.json');old=read(path) if path.exists() else {}
    if old.get('fingerprint')==fingerprint and (old.get('status')=='complete' or old.get('failures',0)>3):return
    attempts=old.get('failures',0) if old.get('fingerprint')==fingerprint else 0
    try:
        fn();write(path,dict(status='complete',fingerprint=fingerprint,finished_unix=time.time(),failures=attempts))
    except Exception as exc:
        attempts+=1;write(path,dict(status='待人工' if attempts>3 else 'retry_pending',fingerprint=fingerprint,failures=attempts,error=repr(exc)))

def run_once():
    runs=available();summary=[]
    for ident,p in sorted(runs.items()):
        cfg=read(p/'config.json') if (p/'config.json').exists() else read(p/'launch_config.json')
        receipt=read(p/'verification.json');summary.append(dict(id=ident,seed=cfg.get('seed'),method=cfg.get('method','sft' if ident=='F03' else None),model=cfg.get('model'),evaluation=receipt.get('evaluation'),run=str(p)))
        metrics=p/'train_metrics.jsonl'
        if metrics.exists():
            def build(p=p,ident=ident,metrics=metrics):
                value=diagnostics([json.loads(l) for l in metrics.read_text().splitlines()])
                value['validation']={q.stem:read(q) for q in sorted(p.glob('validation_*.json'))}
                if (p/'additional_metrics.json').exists():value['additional_metrics']=read(p/'additional_metrics.json')
                write(OUT/'runs'/(ident+'.json'),value)
            guarded(ident+'_curves',digest(p/'verification.json'),build)
            def plot(ident=ident):
                subprocess.run(['/data/venvs/q35-reporting/bin/python',str(Path(__file__).with_name('plot_q35_report.py')),
                                '--data',str(OUT/'runs'/(ident+'.json')),'--output',str(OUT/'plots'/ident),'--title',ident],check=True)
            if (OUT/'runs'/(ident+'.json')).exists():guarded(ident+'_plots',digest(OUT/'runs'/(ident+'.json')),plot)
    for left,right in PAIRS:
        if left not in runs or right not in runs:continue
        key=left+'_minus_'+right
        fingerprint=digest(runs[left]/'verification.json')+digest(runs[right]/'verification.json')
        guarded(key,fingerprint,lambda left=left,right=right,key=key:write(OUT/'paired'/(key+'.json'),report(runs[left],runs[right])))
    grouped={}
    for row in summary:
        if row['id'] in ['F01','F03','S01','S02','H01','H02']:continue
        key=(row['id'][0],row['method'],row['model']);grouped.setdefault(key,[]).append(row)
    aggregates=[]
    for (environment,method,model),values in grouped.items():
        means={}
        for name in ['main100','supplemental500','filtered409','map8']:
            if all(name in (v['evaluation'] or {}) for v in values):
                means[name]={metric:float(np.mean([v['evaluation'][name][metric] for v in values])) for metric in ['success_rate','pass_at_4']}
        aggregates.append(dict(environment=environment,method=method,model=model,seeds=[v['seed'] for v in values],experiments=[v['id'] for v in values],means=means,paired_intervals='see paired/; do not interpret seed means as paired confidence intervals'))
    write(OUT/'summary.json',dict(updated_unix=time.time(),completed_runs=summary,seed_aggregates=aggregates,paired_expected=PAIRS,reporting_only=True,all_P0_complete=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        run_once()
        if not a.watch:break
        time.sleep(60)
