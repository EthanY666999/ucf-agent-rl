import gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
root=Path('/home/vipuser/decision-point-ucf-v3-20261110/code');sys.path[:0]=[str(root),str(root/'ucf')]
from transformers import AutoTokenizer
from first_fence import locate_token_prefix
from ucf_weights import generation_end
tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
cases=[]
for environment in ['S','F']:
 base=Path('/data/runs/ucf_v3_control_20261110/gpu_acceptance')/environment
 selected={ (x['task_id'],x['j'],x['turn']):x for x in [json.loads(l) for l in (base/'ucf_logs/select_0001.jsonl').read_text().splitlines()] if x['selected'] }
 diffs=json.loads((base/'sampling.json').read_text())['stats']['probe_original_logprob_differences'];selected_keys=list(selected);assert len(diffs)==len(selected_keys);diffs=dict(zip(selected_keys,diffs))
 with gzip.open(base/'ucf_logs/raw_leaves_0001.jsonl.gz','rt') as f:
  for line in f:
   d=json.loads(line)
   if d['kind']!='initial':continue
   b=SimpleNamespace(**d);starts=[i for i,m in enumerate(b.mask) if m and (not i or not b.mask[i-1])]
   for turn,start in enumerate(starts,1):
    end=generation_end(b,start);bd=locate_token_prefix(b.ids,start,end,tok)
    if bd is None or bd>=end:continue
    key=(b.task_id,int(b.lineage),turn)
    cases.append(dict(environment=environment,uid=b.task['uid'],task_id=b.task_id,lineage=b.lineage,turn=turn,prefix=b.ids[:bd],prefix_length=bd,original_token=b.ids[bd],original_token_text=tok.decode([b.ids[bd]]),original_logprob=b.old_logprobs[bd],selected=key in selected,smoke_difference=diffs.get(key)))
out=Path('/home/vipuser/decision-point-ucf-v3-20261110/storage/probe_cases.json');out.write_text(json.dumps(cases)+'\n')
print(json.dumps(dict(cases=len(cases),bytes=out.stat().st_size)))
for x in cases:
 if x['selected']:print(json.dumps({k:v for k,v in x.items() if k!='prefix'}))
