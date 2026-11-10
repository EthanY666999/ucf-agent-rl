"""Historical raw trajectory text replay; no parser-match rate assumed zero."""
import argparse,gzip,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from response_protocol import first_block

def run(source,output):
 counts=dict(trajectories=0,turns=0,first_fence_turns=0,complete_blocks=0,first_fence_not_complete_block_start=0)
 hashes={};files=sorted((source/'ucf_logs').glob('raw_leaves_*.jsonl.gz')) or sorted(source.glob('rollouts_*.jsonl.gz'))
 bound=json.loads((source/'status.json').read_text()).get('step',400)
 for f in files:
  if int(f.name.split('_')[-1].split('.')[0])>bound:continue
  hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
  for line in gzip.open(f,'rb'):
   row=json.loads(line)
   if row.get('kind','initial')!='initial':continue
   counts['trajectories']+=1
   for tr in row['trace']:
    text=tr['response'];m=first_block(text);first=text.find('```');counts['turns']+=1;counts['first_fence_turns']+=first>=0
    if m:counts['complete_blocks']+=1;counts['first_fence_not_complete_block_start']+=first!=m.start()
 r=dict(passed=True,source=str(source),through_step=bound,counts=counts,fraction_among_complete_blocks=counts['first_fence_not_complete_block_start']/max(1,counts['complete_blocks']),fraction_among_fenced_turns=counts['first_fence_not_complete_block_start']/max(1,counts['first_fence_turns']),raw_sha256=hashes,uses_test=False,historical_reference_only=True)
 output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='raw_sha256'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run,a.output)
