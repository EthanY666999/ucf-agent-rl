"""Compare original decode scores with single/cached/batched prefill scoring."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cases',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
os.fstat(int(os.environ['UCF_INHERITED_GPU_LOCK_FD']))
root=Path('/home/vipuser/decision-point-ucf-v3-20261110/code');sys.path[:0]=[str(root),str(root/'ucf')]
import numpy as np
import train
from vllm import SamplingParams
from ucf_probe import requests
from v3_sampling import full_vector_from_output
cfg=json.loads(Path(a.config).read_text());cases=json.loads(Path(a.cases).read_text());a.output.mkdir(parents=True,exist_ok=True)
started=time.time();actor,ref,engine,tok=train.build_models(cfg,42);vocab=engine.llm_engine.model_config.get_vocab_size()
params=lambda:SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,logprobs=-1,flat_logprobs=True,seed=42)
results=[];costs=[]
for i,x in enumerate(cases):
 engine.llm_engine.reset_prefix_cache()
 vectors=[]
 for repetition in range(2):
  outputs,cost=requests(engine,[(i,x['prefix'],params())]);costs.append(cost)
  lp,z=full_vector_from_output(outputs[i].outputs[0],vocab);vectors.append(lp)
 row={k:v for k,v in x.items() if k!='prefix'};t=x['original_token']
 row.update(cold_logprob=float(vectors[0][t]),cached_logprob=float(vectors[1][t]),cold_delta=float(vectors[0][t]-x['original_logprob']),cached_delta=float(vectors[1][t]-x['original_logprob']),cache_total_variation=float(np.abs(np.exp(vectors[0])-np.exp(vectors[1])).sum()/2))
 results.append(row);del outputs,vectors
 with (a.output/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
# Bounded groups avoid retaining every full-vocabulary vector concurrently.
engine.llm_engine.reset_prefix_cache()
for start in range(0,len(cases),8):
 outputs,cost=requests(engine,[(i,cases[i]['prefix'],params()) for i in range(start,min(start+8,len(cases)))]);costs.append(cost)
 for i,o in outputs.items():
  lp,z=full_vector_from_output(o.outputs[0],vocab);results[i]['batch_delta']=float(lp[cases[i]['original_token']]-cases[i]['original_logprob']);del lp
 del outputs
report=dict(seconds=time.time()-started,cases=len(cases),selected_cases=sum(x['selected'] for x in cases),results=results,cost={k:sum(x[k] for x in costs) for k in ['input_tokens','generated_tokens','seconds']},interpretation='Numerical diagnosis only; original smoke failures are not erased',thresholds=dict(max_absolute_delta=.125,mean_absolute_delta=.02))
report['summaries']={environment:{mode:dict(mean_abs=float(np.mean([abs(x[mode+'_delta']) for x in results if x['environment']==environment])),max_abs=max(abs(x[mode+'_delta']) for x in results if x['environment']==environment)) for mode in ['cold','cached','batch']} for environment in ['S','F']}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summaries']),flush=True)
