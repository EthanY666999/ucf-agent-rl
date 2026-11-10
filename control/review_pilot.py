"""CPU-only fixed-validation review. Never opens final/supplemental test data."""
import argparse,hashlib,json,math,sys,time
from collections import defaultdict
from pathlib import Path

ROOT=Path('/home/vipuser/decision-point-ucf-v3-20261110/code')
sys.path[:0]=[str(ROOT/'scripts')]
from validation_recovery import validation_complete
from paired_report import task_scores,bootstrap

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_validation(run,step):
 cfg=read(run/'config.json');data=Path(cfg['dataset_path'])
 assert sha(data)==cfg['dataset_sha256']
 tasks=read(data)['validation'];p=run/f'validation_{step:04d}.json'
 assert validation_complete(p,tasks),f'Unverified/incomplete fixed validation: {p}'
 scores=task_scores([json.loads(l) for l in p.with_suffix('.jsonl').read_text().splitlines()])
 return cfg,read(p),scores,dict(summary=sha(p),records=sha(p.with_suffix('.jsonl')))

def compare(a,b):
 assert set(a)==set(b)
 result=dict(overall=bootstrap(a,b,list(a)))
 if all('shortest_path_length' in x['task'] for x in a.values()):
  strata=defaultdict(list)
  for uid,x in a.items():strata[x['task']['shortest_path_length']].append(uid)
  result['by_shortest_path']={str(k):dict(bootstrap(a,b,ids),
    left_success_pct=100*sum(a[u]['success'] for u in ids)/len(ids),
    right_success_pct=100*sum(b[u]['success'] for u in ids)/len(ids)) for k,ids in sorted(strata.items())}
 return result

def review(run,step,comparison=None):
 cfg,current,left,hashes=load_validation(run,step)
 origin=read(run/'initialization.json') if (run/'initialization.json').exists() else dict(kind='fresh_raw_model',initial_step=0,task_sft=False)
 base=origin['initial_step']
 _,initial,start,initial_hashes=load_validation(run,base)
 metrics=[json.loads(l) for l in (run/'train_metrics.jsonl').read_text().splitlines()]
 metrics=[r for r in metrics if r['step']<=step]
 assert [r['step'] for r in metrics]==list(range(base+1,step+1))
 keys=['loss','kl','grad_norm','step_seconds','max_parameter_change']
 finite=all(math.isfinite(r[k]) for r in metrics for k in keys)
 window=metrics[-25:]
 diagnostics={k:dict(mean=sum(r[k] for r in window)/len(window),maximum=max(r[k] for r in window)) for k in keys+['success_rate','mean_response_tokens','mean_turns']}
 result=dict(unix=time.time(),run=str(run),step=step,final_test_used=False,
  initialization=origin,continuation_updates=step-base,
  configured_reference_model_is_raw=cfg['model']=='/data/models/Qwen3.5-2B',initial_validation=initial,
  current_validation=current,raw_validation_sha256=hashes,initial_validation_sha256=initial_hashes,
  improvement_over_initialization=compare(left,start),finite_training=finite,
  updates_with_nonzero_parameter_change=sum(r['max_parameter_change']>0 for r in metrics),
  last25_training=diagnostics,
  caveat='Exploratory validation with repeated looks; task bootstrap intervals are descriptive, not a held-out-test claim. Baseline learning alone is not UCF gain.')
 if comparison:
  cfg2,summary2,right,hashes2=load_validation(comparison,step)
  origin2=read(comparison/'initialization.json') if (comparison/'initialization.json').exists() else dict(kind='fresh_raw_model',initial_step=0,task_sft=False)
  assert origin['kind']==origin2['kind'] and origin['initial_step']==origin2['initial_step'],'Unmatched starting point'
  if base:
   assert origin['source_checkpoint_sha256']==origin2['source_checkpoint_sha256']
   assert origin['tensor_optimizer_rng_sha256']==origin2['tensor_optimizer_rng_sha256']
  matched=['model','seed','dataset_sha256','frozen_maps_sha256','max_turns','max_response_tokens',
   'response_protocol','initial_branches','flat_group_size','temperature','top_p','top_k',
   'eval_samples','eval_temperature','eval_top_p','eval_top_k','learning_rate','kl_coefficient',
   'clip_low','clip_high','train_batch_size','mini_batch_tasks','optimizer','vllm_batch_invariant',
   'vllm_batch_invariant_scoped']
  assert all(cfg.get(k)==cfg2.get(k) for k in matched),'Unmatched training/evaluation protocol'
  if cfg.get('vllm_batch_invariant_scoped'):
   numerical=read(run/'numerical_runtime.json');numerical2=read(comparison/'numerical_runtime.json')
   fields=['protocol','torch','vllm','attention_backend','helper_sha256']
   assert all(numerical.get(k)==numerical2.get(k) for k in fields),'Unmatched numerical runtime'
  result['paired_comparison']=dict(right_run=str(comparison),right_validation=summary2,
   right_raw_validation_sha256=hashes2,direction='requested run minus comparison',
   probability_source_left=cfg.get('ucf_probability_source','prefix_replay_probe'),
   probability_source_right=cfg2.get('ucf_probability_source','not_applicable_to_grpo'),
   **compare(left,right))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--step',required=True,type=int);p.add_argument('--comparison',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 assert a.step>0
 result=review(a.run,a.step,a.comparison);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(step=a.step,validation=result['current_validation']['success_rate_pct'],delta=result['improvement_over_initialization']['overall']['success'],finite=result['finite_training'])))
