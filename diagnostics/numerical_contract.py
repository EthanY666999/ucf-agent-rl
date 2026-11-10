"""Standalone GPU diagnostic. Does not patch the active runtime or environment."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--batch-invariant',action='store_true');a=p.parse_args()
os.fstat(int(os.environ['UCF_INHERITED_GPU_LOCK_FD']))
os.environ['VLLM_BATCH_INVARIANT']='1' if a.batch_invariant else '0'
root=Path('/home/vipuser/decision-point-ucf-v3-20261110/code');sys.path[:0]=[str(root),str(root/'ucf'),str(root/'scripts')]
import train
import patr_core as core
import v3_gpu_acceptance
if a.batch_invariant:
 import vllm
 original_llm=vllm.LLM
 def diagnostic_llm(*args,**kwargs):
  # v0.19.1 checks this before automatic backend selection runs.
  kwargs['attention_config']={'backend':'FLASH_ATTN'}
  return original_llm(*args,**kwargs)
 vllm.LLM=diagnostic_llm
original=train.build_models
def build(cfg,seed):
 result=original(cfg,seed);engine=result[2];ec=engine.llm_engine.vllm_config
 core.atomic_json(a.output/'numerical_backend.json',dict(unix=time.time(),batch_invariant=a.batch_invariant,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_config={k:str(getattr(ec.cache_config,k,None)) for k in ("cache_dtype","mamba_cache_dtype","mamba_ssm_cache_dtype","mamba_cache_mode","enable_prefix_caching","gpu_memory_utilization")},vllm_config_model_dtype=str(ec.model_config.dtype),source='isolated diagnostic; active runtime unmodified'))
 return result
train.build_models=build
sys.argv=['v3_gpu_acceptance.py','--config',a.config,'--output',str(a.output)]
v3_gpu_acceptance.main()
