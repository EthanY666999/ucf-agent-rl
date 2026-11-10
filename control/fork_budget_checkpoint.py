"""Explicit RL-warmstart fork; never relax train.py's strict resume guard."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

STATE=Path('/data/runs/ucf_v3_budget10240_20261110')
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def payload_sha(state):
    """Content identity, independent of PyTorch serialization and config names."""
    import torch
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            value=value.detach().cpu().contiguous()
            h.update(str((str(value.dtype),tuple(value.shape))).encode())
            h.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(value,dict):
            h.update(b'dict')
            for k in sorted(value,key=repr):visit(k);visit(value[k])
        elif isinstance(value,(tuple,list)):
            h.update(type(value).__name__.encode())
            for v in value:visit(v)
        else:h.update(repr(value).encode()+b'\0')
    for key in ('model','optimizer','rng','torch_rng','cuda_rng','python_rng','step','elapsed_seconds'):
        visit(key);visit(state.get(key))
    return h.hexdigest()
def continuation_config(source,method,out):
    assert method in ('ucf','grpo')
    assert source['max_response_tokens']==4096 and source['ucf_v3']
    cfg=dict(source,max_response_tokens=10240,ucf_probability_source='original_sampling_logits',run_label=out.name)
    if method=='ucf':cfg['ucf_log_dir']=str(out/'ucf_logs')
    else:cfg.pop('ucf_log_dir',None)
    allowed={'max_response_tokens','ucf_probability_source','run_label','ucf_log_dir'}
    assert {k:v for k,v in source.items() if k not in allowed}=={k:v for k,v in cfg.items() if k not in allowed}
    return cfg

def main():
    import fcntl
    import torch
    locks=[]
    for p in ('/data/runs/queue_q35_hard/supervisor.lock','/data/runs/gpu80.lock'):
        f=open(p,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    proof=read(STATE/'admission/acceptance.json');assert proof['passed'] and proof['max_response_tokens']==10240
    origin=read(STATE/'source_checkpoint.json');cp=Path(origin['path'])
    assert sha(cp)==origin['sha256']==proof['checkpoint_sha256']
    status=read(cp.parent/'status.json');assert status['status']=='paused_checkpointed' and status['step']==16
    src=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    assert src['step']==16
    digest=payload_sha(src)
    outputs=[('grpo',Path('/data/runs/ucf_v3_budget10240_runs/V3_S_grpo_budget10240_from16_seed42')),
        ('ucf',Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_budget10240_from16_seed42'))]
    records=[]
    for method,out in outputs:
        out.mkdir(parents=True,exist_ok=True)
        target=out/'checkpoint.pt'
        cfg=continuation_config(src['config'],method,out)
        if target.exists():
            existing=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
            assert existing['config']==cfg and payload_sha(existing)==digest,'Existing continuation is different; do not overwrite'
            del existing
        else:
            assert shutil.disk_usage(out).free>cp.stat().st_size+8*2**30,'Insufficient reserve for a new independent checkpoint'
            fork=dict(src,config=cfg)
            tmp=out/'checkpoint.fork.tmp'
            assert not tmp.exists(),'Inspect incomplete prior fork before replacing it'
            torch.save(fork,tmp)
            with tmp.open('rb') as f:os.fsync(f.fileno())
            check=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
            assert check['config']==cfg and payload_sha(check)==digest
            del check
            tmp.replace(target)
        identity=dict(kind='shared_rl_warmstart_budget_exploration',initial_step=16,
            source_run=str(cp.parent),source_checkpoint_sha256=origin['sha256'],
            tensor_optimizer_rng_sha256=digest,reference_model=cfg['model'],task_sft=False,
            prior_training_method='ucf',prior_training_steps=16,prior_max_response_tokens=4096,
            continuation_max_response_tokens=10240,method=method,
            not_a_fresh_raw_start_comparison=True,unix=time.time())
        if method=='ucf':
            controller=Path(src['config']['ucf_log_dir'])/'controller_0016.json'
            assert read(controller)['last_step']==16
            (out/'ucf_logs').mkdir(exist_ok=True)
            shutil.copy2(controller,out/'ucf_logs/controller_0016.json')
            shutil.copy2(controller,out/'ucf_logs/controller.json')
            identity['imported_controller_sha256']=sha(controller)
            identity['controller_note']='Historical uniform-selector boundary receipt; next selection is freshly calibrated from the new batch'
        atomic(out/'initialization.json',identity)
        atomic(out/'continuation_config.json',cfg)
        atomic(out/'config.json',dict(cfg,method=method))
        atomic(out/'status.json',dict(status='paused_checkpointed',step=16,method=method,total_steps=400,
            reason='explicitly imported identical RL checkpoint for paired budget continuation'))
        for suffix in ('.json','.jsonl'):
            name='validation_0016'+suffix
            shutil.copy2(STATE/'admission'/name,out/name)
        records.append(dict(method=method,path=str(target),bytes=target.stat().st_size,sha256=sha(target),
            config_path=str(out/'continuation_config.json'),config_sha256=sha(out/'continuation_config.json'),
            tensor_optimizer_rng_sha256=digest))
    atomic(STATE/'checkpoint_forks.json',dict(source=origin,outputs=records,
        tensor_optimizer_rng_equal=True,source_checkpoint_preserved=True,unix=time.time()))
    atomic(STATE/'planned_duplicate_removal.json',dict(path=str(cp),bytes=cp.stat().st_size,sha256=origin['sha256'],
        reason='Two independently verified continuations preserve all model, optimizer and RNG state at step16',
        replacements=records,records_logs_source_config_retained=True,
        execute_only_after_manifest_is_published=True,unix=time.time()))
    print(json.dumps(dict(forked=len(records),source_checkpoint_preserved=True)))

if __name__=='__main__':main()
