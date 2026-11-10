"""Isolated real-GPU admission for scoped inference kernels; no training edits."""
import argparse, gzip, hashlib, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--batch-invariant',action='store_true');a=p.parse_args()
os.fstat(int(os.environ['UCF_INHERITED_GPU_LOCK_FD']))
os.environ['VLLM_BATCH_INVARIANT']='1'
root=Path('/home/vipuser/decision-point-ucf-v3-20261110/code');sys.path[:0]=[str(root),str(root/'ucf'),str(root/'scripts')]
import train, torch, vllm
import patr_core as core
import v3_gpu_acceptance
from scoped_inference import configure_train, tiny_native_gradients
scope=configure_train(train,torch,vllm)
def native_checks():
 return tiny_native_gradients(torch)+tiny_native_gradients(torch,autocast=True)
reference=native_checks()
checks=[]
for cycle in range(3):
 scope.inference()
 with torch.no_grad():
  x=torch.ones((2,8),device='cuda');y=torch.ones((8,5),device='cuda');z=x@y
  assert torch.equal(z,torch.full((2,5),8.,device='cuda'))
 scope.training();grads=native_checks()
 difference=max(float((a-b).abs().max()) for a,b in zip(grads,reference))
 assert difference==0.,difference
 checks.append(dict(cycle=cycle,gradient_max_difference=difference))

saved={}
scoped_build=train.build_models
def build(cfg,seed):
 result=scoped_build(cfg,seed);saved['engine']=result[2]
 return result
train.build_models=build
sys.argv=['v3_gpu_acceptance.py','--config',a.config,'--output',str(a.output)]
v3_gpu_acceptance.main()
# Verify returning to inference after a real update, then native backward again.
engine=saved['engine']
from vllm import SamplingParams
from ucf_probe import requests
with gzip.open(a.output/'ucf_logs/raw_leaves_0001.jsonl.gz','rt') as f:
 recorded_prefix=json.loads(next(f))['ids'][:64]
requests(engine,[('post_update',recorded_prefix,SamplingParams(max_tokens=1,temperature=1.,seed=42))])
scope.training();after=native_checks()
assert all(torch.equal(x,y) for x,y in zip(reference,after))
core.atomic_json(a.output/'scope_acceptance.json',dict(passed=True,unix=time.time(),
 software=dict(torch=torch.__version__,vllm=vllm.__version__),
 helper_sha256=hashlib.sha256(Path(__file__).with_name('scoped_inference.py').read_bytes()).hexdigest(),
 wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 native_gradient_checks=checks,native_fp32_and_bf16_autocast_checked=True,post_update_inference_then_backward=True,
 transitions=scope.transitions,installed_files_modified=False,
 interpretation='Engineering-only admission; requires a separately recorded matched numerical protocol for training'))
