"""Verify completed runs and retain analysis records before checkpoint pruning."""
import hashlib
import json
import math
import time
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def summarize(records,n):
    groups={}
    for r in records:
        assert r['reward'] in [0,1]
        groups.setdefault(r['task']['uid'],[]).append(r)
    assert len(groups)==n and all(len(rs)==4 for rs in groups.values())
    return dict(n_tasks=n,success_rate=sum(r['reward'] for r in records)/len(records),
                pass_at_4=sum(any(r['reward'] for r in rs) for rs in groups.values())/n)


def verify(out,sft=False,prune_checkpoint=False):
    out=Path(out);evaluation=out/'evaluation' if sft else out
    train=out/'sft' if sft else out
    assert read(evaluation/'status.json')['status'] in ['complete','evaluation_complete']
    if sft:assert read(train/'status.json')['status']=='complete'
    expected_steps=63 if sft else 400
    metrics=rows(train/('metrics.jsonl' if sft else 'train_metrics.jsonl'))
    by_step={r['step']:r for r in metrics}
    assert set(by_step)==set(range(1,expected_steps+1))
    for r in metrics:
        for k in ['loss','grad_norm','kl']:
            if k in r:assert math.isfinite(r[k]),(r['step'],k)
    cfg=read(out/'launch_config.json') if (out/'launch_config.json').exists() else read(out/'config.json')
    hard=cfg.get('environment') in ['flh','sokoban']
    evaluations={}
    for label,file,n in ([('main100','final_test.jsonl',100)] if hard else [('main100','final_test.jsonl',100),('map8','map8.jsonl',100)]):
        evaluations[label]=summarize(rows(evaluation/file),n)
    supplemental=[]
    for i in range(1,6):supplemental.extend(rows(evaluation/f'supplemental500/batch{i}.jsonl'))
    evaluations['supplemental500']=summarize(supplemental,500)
    root=Path(__file__).resolve().parents[1]
    if not hard:
        uids={r['uid'] for r in read(Path(cfg.get('eval_sets_dir',root/'eval_sets'))/'filtered409/tasks.json')}
        evaluations['filtered409']=summarize([r for r in supplemental if r['task']['uid'] in uids],409)
    for label in (['supplemental500'] if hard else ['supplemental500','filtered409']):
        recorded=read(evaluation/('supplemental500/summary.json' if label=='supplemental500' else 'filtered409.json'))
        for key in ['success_rate','pass_at_4']:assert abs(recorded[key]-evaluations[label][key])<1e-12
    if not sft:
        for step in range(0,401,25):assert (evaluation/f'validation_{step:04d}.json').exists()
    from hard_metrics import completed_metrics
    completed_metrics(evaluation,cfg)
    model=train/'final_model';shards=sorted(model.glob('*.safetensors'));assert shards
    from safetensors import safe_open
    for shard in shards:
        with safe_open(shard,framework='numpy') as f:assert list(f.keys())
    assert (model/'config.json').exists() and (model/'tokenizer_config.json').exists()
    index=model/'model.safetensors.index.json'
    if index.exists():assert set(read(index)['weight_map'].values())=={p.name for p in shards}
    evidence={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl'] and p.name not in ['verification.json','run_report.json']}
    receipt=dict(verified_unix=time.time(),passed=True,steps=expected_steps,
                 metrics_rows=len(metrics),duplicate_metric_rows=len(metrics)-len(by_step),
                 evaluation=evaluations,record_sha256=evidence,
                 final_model_sha256={p.name:sha(p) for p in shards},
                 final_model_retained=True,rollouts_retained=True,github_publication='pending')
    atomic(out/'verification.json',receipt)
    atomic(out/'run_report.json',dict(evaluation=evaluations,steps=expected_steps,
            last_training=by_step[expected_steps],paired_bootstrap='pending paired comparator completion',
            verified=True,selection_uses_test=False))
    cp=train/'checkpoint.pt'
    if prune_checkpoint and cp.exists():
        assert not cp.is_symlink(),'Never prune relocated checkpoints implicitly'
        # All per-task results, diagnostics, configuration and final weights stay.
        deletion=dict(unix=time.time(),path=str(cp),bytes=cp.stat().st_size,
                      reason='Final evaluation and all 400/63 training steps verified; records and final_model retained',
                      verification_sha256=sha(out/'verification.json'))
        atomic(out/'checkpoint_pruning.json',dict(deletion,status='planned'))
        cp.unlink();atomic(out/'checkpoint_pruning.json',dict(deletion,status='complete'))
    return receipt
