"""Training-only CPU audit of first-complete-block protocol, no model weights."""
import argparse,gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
from response_protocol import first_block,parse_first_block,NAME
from first_fence import segments
try:
 import orjson
 loads=orjson.loads
except ImportError:loads=json.loads

def audit(run,output,new_protocol=False,max_step=None):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
 cfg=loads((run/'config.json').read_bytes())
 if new_protocol:
  assert cfg.get('response_protocol')==NAME
  expected={t['uid']:t for t in loads(Path(cfg['dataset_path']).read_bytes())['train']}
 files=sorted((run/'ucf_logs').glob('raw_leaves_*.jsonl.gz')) or sorted(run.glob('rollouts_*.jsonl.gz'))
 assert files
 counts=Counter();hashes={};steps=[]
 fence=tok.encode('```',add_special_tokens=False)[0]
 for f in files:
  step=int(f.name.split('_')[-1].split('.')[0])
  if max_step is not None and step>max_step:continue
  steps.append(step);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
  groups={}
  with gzip.open(f,'rb') as fp:
   for line in fp:
    r=loads(line)
    if r.get('kind','initial')!='initial':continue
    if new_protocol:
     assert r['task']==expected[r['task']['uid']], 'Frozen training task changed'
     group=groups.setdefault(r['task']['uid'],set());assert str(r['lineage']) not in group;group.add(str(r['lineage']))
    spans=list(segments(r['mask']));assert len(spans)==len(r['trace'])
    counts['trajectories']+=1
    for (start,end),tr in zip(spans,r['trace']):
     counts['turns']+=1;text=tr['response'];m=first_block(text)
     ids=r['ids'][start:end]
     if m:
      counts['complete_blocks']+=1
      counts['first_fence_not_first_complete_block_start']+=m.start()!=text.find('```')
      counts['boundary_token_extra_characters']+=bool(tr.get('protocol_cutoff',{}).get('boundary_token_overhang'))
      counts['changed_text_after_cut']+=bool(text[m.end():])
      # Find exact sampled token endpoint; preserve all original IDs/logprobs.
      exact=None
      for j,t in enumerate(ids):
       if t==fence:
        prefix=tok.decode(ids[:j+1],skip_special_tokens=True)
        if prefix==text[:m.end()]:exact=j+1;break
      if exact is None:
       # Closing fence can be a different token, e.g. leading-space fence.
       for j in range(len(ids)):
        if tok.decode(ids[:j+1],skip_special_tokens=True)==text[:m.end()]:exact=j+1;break
      counts['closing_endpoint_not_token_aligned']+=exact is None
      boundary=exact
      if boundary is None:
       for j in range(len(ids)):
        decoded=tok.decode(ids[:j+1],skip_special_tokens=True)
        if decoded.startswith(text[:m.end()]):boundary=j+1;break
      counts['closing_block_without_token_boundary']+=boundary is None
      if exact is not None:
       tail=ids[exact:]
       counts['changed_sampled_token_sequence_after_cut']+=bool(tail)
       counts['discarded_only_eos']+=bool(tail) and all(t==tok.eos_token_id for t in tail)
       counts['discarded_non_eos_tokens']+=sum(t!=tok.eos_token_id for t in tail)
      counts['new_parser_differs_from_recorded_action']+=int(parse_first_block(None,text)[1])!=(tr.get('action') or 0)
     first=text.find('```')
     if first>=0:
      counts['first_fence_turns']+=1
      # Semantic action after optional whitespace, closing the FIRST fence.
      # Other fenced text is residual/invalid, not silently replaced by a later block.
      initial=first_block(text[first:])
      initial_action=int(parse_first_block(None,text[first:])[1]) if initial and initial.start()==0 else 0
      executed=tr.get('action') or 0
      counts['first_fence_parse_mismatch']+=initial_action!=executed
     if new_protocol:
      if m:
       assert boundary==len(ids) and text==tok.decode(ids,skip_special_tokens=True), 'Post-boundary tokens retained/trained'
       assert tr['protocol_cutoff']['boundary_token_overhang']==text[m.end():]
       assert tr.get('protocol_cutoff',{}).get('protocol')==NAME
  if new_protocol:assert len(groups)==64 and all(g=={'0','1','2','3'} for g in groups.values())
  if len(steps)%50==0:print(json.dumps({'audited_steps':len(steps)}),flush=True)
 if new_protocol:assert steps==list(range(1,401)), 'Require all 400 complete training batches'
 ratio=counts['first_fence_parse_mismatch']/max(1,counts['first_fence_turns'])
 passed=(not new_protocol or ratio<=.02) and counts['closing_block_without_token_boundary']==0
 result=dict(passed=passed,training_only=True,source=str(run),source_experiment=run.name.split('_')[0],
   complete_steps=max(steps),audited_step_count=len(steps),response_protocol=cfg.get('response_protocol','legacy_last_block'),
   counts=dict(counts),first_fence_not_first_complete_block_fraction=counts['first_fence_not_first_complete_block_start']/max(1,counts['complete_blocks']),boundary_token_extra_fraction=counts['boundary_token_extra_characters']/max(1,counts['complete_blocks']),mismatch_fraction=ratio,mismatch_threshold=.02,
   mismatch_definition='action in the FIRST opened fence after whitespace and before its closing fence vs actual executed action; invalid block=0',
   boundary_unit='first sampled token completing the action block',hypothetical_cut_only=not new_protocol,SFT_runtime_unchanged=not new_protocol and cfg['model']!='/data/models/Qwen3.5-2B',
   expectation_method='LOO c requires all action-token probabilities. GRPO raw top-1 logs cannot recover exact entropy/rho; no numeric expectation fabricated.',
   expected_forks_per_task=None,record_sha256=hashes)
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--new-protocol',action='store_true');p.add_argument('--max-step',type=int);a=p.parse_args()
 r=audit(a.run,a.output,a.new_protocol,a.max_step);print(json.dumps({k:v for k,v in r.items() if k!='record_sha256'}))
