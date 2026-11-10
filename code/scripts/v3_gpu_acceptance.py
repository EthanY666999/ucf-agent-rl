"""Train-only V3 GPU contract; no checkpoint, model export or test-set use."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ucf'),str(ROOT/'ucf/scripts')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    # Lock is inherited from the boundary manager, including if it crashes.
    os.fstat(int(os.environ['UCF_INHERITED_GPU_LOCK_FD']))
    import numpy as np
    import torch
    import train
    import patr_core as core
    from ucf_v3 import make_collect
    from ucf_weights import training_weights_ucf
    from accept_ucf import component_grad
    from scripts.q35_thinking_policy import check_prompt_contract
    cfg=json.loads(Path(args.config).read_text())
    assert cfg['ucf_v3'] and cfg['model']=='/data/models/Qwen3.5-2B'
    assert cfg['repeat_limit']>cfg['max_turns']
    cfg=dict(cfg,train_batch_size=4,mini_batch_tasks=4,ucf_log_dir=str(args.output/'ucf_logs'))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    core.atomic_json(out/'config.json',cfg)
    core.atomic_json(out/'status.json',dict(status='gpu_contract',current_step=1,pid=os.getpid()))
    os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
    started=time.monotonic()
    actor,ref,engine,tok=train.build_models(cfg,42)
    prompt=check_prompt_contract(tok)
    tasks=core.make_datasets(cfg)['train']
    ids=np.random.default_rng(np.random.SeedSequence([42,1,1701])).choice(len(tasks),64,replace=False)[:4]
    tasks=[tasks[int(i)] for i in ids]
    collector=make_collect(core,core.collect)
    branches,stats,events=collector(tasks,engine,tok,cfg,'ucf',np.random.default_rng(42))
    train.write_rollouts(out/'rollouts.jsonl.gz',branches,tasks)
    core.atomic_json(out/'sampling.json',dict(stats=stats,events=events,train_only=True))
    assert stats['branch_events']>0 and len([b for b in branches if hasattr(b,'ucf')])>0
    assert all(abs(x)<1e-4 for x in stats['full_vocabulary_log_normalizers'])
    differences=[abs(x) for x in stats['probe_original_logprob_differences']]
    assert max(differences)<=.125 and np.mean(differences)<=.02, ('prefill/decode drift',differences)
    assert stats['cf_fallback_fraction']<1., 'All selected complements are empty'
    assert all(math.isfinite(lp) for b in branches for lp in b.old_logprobs)
    train.training_weights=training_weights_ucf
    components={p:component_grad(actor,ref,branches,cfg,tok,p) for p in ('A','B')}
    opt=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    update=train.update(actor,ref,opt,branches,cfg,'ucf',tok,np.random.default_rng(42),len(tasks))
    assert all(math.isfinite(update[k]) for k in ('loss','grad_norm','kl'))
    # A train-only small batch may have no nonzero outcome advantage. In that
    # case a finite zero update is a data diagnostic, not invented learning.
    train.sync_weights(actor,engine)
    core.atomic_json(out/'acceptance.json',dict(passed=True,train_only=True,task_ids=[t['uid'] for t in tasks],
        model_weights_retained=False,steps=1,seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,prompt_contract=prompt,
        sampling=stats,components=components,update=update,
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        interpretation='Technical GPU contract only; not a training quality result'))
    core.atomic_json(out/'status.json',dict(status='gpu_contract_passed',unix=time.time()))


if __name__=='__main__':main()
