"""Isolated train-only admission, after the current stage releases both locks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
os.fstat(int(os.environ['UCF_INHERITED_GPU_LOCK_FD']))
os.environ['VLLM_BATCH_INVARIANT']='1'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'code'),str(root/'code/ucf'),str(root/'code/ucf/scripts'),str(root/'storage')]
import train
import torch
import numpy as np
import vllm
import patr_core as core
import ucf_v3
from original_logits import CaptureSession
from ucf_weights import training_weights_ucf
from accept_ucf import component_grad
from scoped_inference import configure_train,tiny_native_gradients

scope=configure_train(train,torch,vllm)
native=tiny_native_gradients(torch)+tiny_native_gradients(torch,True)
cfg=json.loads(a.config.read_text())
assert cfg['ucf_probability_source']=='original_sampling_logits'
assert cfg['ucf_v3'] and cfg['model']=='/data/models/Qwen3.5-2B'
cfg=dict(cfg,train_batch_size=8,mini_batch_tasks=8,ucf_log_dir=str(a.output/'ucf_logs'))
a.output.mkdir(parents=True,exist_ok=True)
core.atomic_json(a.output/'config.json',cfg)
core.atomic_json(a.output/'status.json',dict(status='capture_gpu_admission',current_step=1,pid=os.getpid()))
os.environ['PATR_PROGRESS_PATH']=str(a.output/'progress.json')
started=time.monotonic()
actor,ref,engine,tok=train.build_models(cfg,42)
tasks_all=core.make_datasets(cfg)['train']
indexes=np.random.default_rng(np.random.SeedSequence([42,1,1701])).choice(len(tasks_all),64,replace=False)[:8]
tasks=[tasks_all[int(i)] for i in indexes]
seed=int(np.random.default_rng(42).integers(0,2**31-1))

# Identical initial sampler, first without observation, then with observation.
# Reset cache in both cases. All seeds and the task ordering remain fixed.
engine.llm_engine.reset_prefix_cache()
off,_,off_cost,_=ucf_v3.candidates_from_initial(core,tasks,engine,tok,cfg,seed)
off_map={(b.task_id,b.lineage):b for b in off}
engine.llm_engine.reset_prefix_cache()
captured=[]
real_retain=CaptureSession.retain
def audit_retain(session,picked):
    for key,(prefix,raw) in session.records.items():
        original=off_map[(key[0],key[1])]
        bd=len(prefix)
        # Includes unselected nodes: do not cherry-pick only small deltas.
        same_prefix=prefix==tuple(original.ids[:bd]) and bd<len(original.ids)
        if same_prefix:
            vector=raw.astype(np.float64);peak=float(vector.max())
            z=peak+math.log(float(np.exp(vector-peak).sum()))
            token=original.ids[bd]
            delta=float(vector[token]-z-original.old_logprobs[bd])
        else:
            z=None;delta=None
        captured.append(dict(key=key,boundary=bd,same_prefix=same_prefix,
                             original_logprob_delta=delta,normalizer=z))
    real_retain(session,picked)
CaptureSession.retain=audit_retain
try:
    branches,stats,events=ucf_v3.make_collect(core,core.collect)(
        tasks,engine,tok,cfg,'ucf',np.random.default_rng(42))
finally:
    CaptureSession.retain=real_retain

originals=[b for b in branches if not hasattr(b,'ucf')]
assert len(originals)==len(off)
comparison=[]
for b in originals:
    previous=off_map[(b.task_id,b.lineage)]
    comparison.append(dict(task_id=b.task_id,lineage=b.lineage,
        ids_equal=b.ids==previous.ids,masks_equal=b.mask==previous.mask,
        logprobs_equal=b.old_logprobs==previous.old_logprobs,
        reward_equal=b.reward==previous.reward,terminal_equal=b.terminal==previous.terminal))
proof=dict(original_trajectories=comparison,all_captured_nodes=captured,
           observer_off_cost=off_cost,statistics=stats)
core.atomic_json(a.output/'observer_noop.json',proof)
train.write_rollouts(a.output/'rollouts.jsonl.gz',branches,tasks)
core.atomic_json(a.output/'sampling.json',dict(stats=stats,events=events,train_only=True))
assert all(all(row[k] for k in ('ids_equal','masks_equal','logprobs_equal','reward_equal','terminal_equal'))
           for row in comparison),'Observer changed original sampling'
assert captured and all(x['same_prefix'] and abs(x['original_logprob_delta'])<=1e-4 for x in captured)
assert stats['branch_events']>0 and stats['cf_fallback_fraction']<1
assert max(map(abs,stats['probe_original_logprob_differences']))<=1e-4
assert stats['original_logits_capture']['pending_vectors']==0
assert stats['probe_cost']['generated_tokens']==stats['probe_cost']['input_tokens']==0

train.training_weights=training_weights_ucf
components={part:component_grad(actor,ref,branches,cfg,tok,part) for part in ('A','B')}
optimizer=torch.optim.AdamW(actor.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
update=train.update(actor,ref,optimizer,branches,cfg,'ucf',tok,np.random.default_rng(42),len(tasks))
assert all(math.isfinite(update[k]) for k in ('loss','grad_norm','kl'))
train.sync_weights(actor,engine)
from vllm import SamplingParams
from ucf_probe import requests
requests(engine,[('post-update',off[0].ids[:64],SamplingParams(max_tokens=1,temperature=1.,seed=42))])
scope.training()
after=tiny_native_gradients(torch)+tiny_native_gradients(torch,True)
assert all(torch.equal(x,y) for x,y in zip(native,after))
result=dict(passed=True,train_only=True,steps=1,seconds=time.monotonic()-started,
    task_ids=[t['uid'] for t in tasks],sampling=stats,components=components,update=update,
    captured_nodes_checked=len(captured),original_sampling_bitwise_unchanged=True,
    max_original_capture_logprob_delta=max(abs(x['original_logprob_delta']) for x in captured),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,model_weights_retained=False,
    config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
    observer_sha256=hashlib.sha256((root/'code/ucf/original_logits.py').read_bytes()).hexdigest(),
    interpretation='Engineering-only probability and no-op admission, not a learning result')
core.atomic_json(a.output/'acceptance.json',result)
core.atomic_json(a.output/'scope_acceptance.json',dict(passed=True,
    helper_sha256=hashlib.sha256((root/'storage/scoped_inference.py').read_bytes()).hexdigest(),
    post_update_inference_then_backward=True,native_fp32_and_bf16_autocast_checked=True,
    installed_files_modified=False,transitions=scope.transitions))
core.atomic_json(a.output/'status.json',dict(status='capture_gpu_admission_passed',unix=time.time()))
print(json.dumps({k:result[k] for k in ('passed','seconds','captured_nodes_checked',
                                      'max_original_capture_logprob_delta','original_sampling_bitwise_unchanged')}))
