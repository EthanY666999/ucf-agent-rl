"""Reconcile UCF controller and evidence to an existing optimizer checkpoint."""
import hashlib,json,re,time
from pathlib import Path

def reconcile_checkpoint(run_root,checkpoint_step):
 root=Path(run_root);logs=root/'ucf_logs';formal=root/'runs/formal/patr_seed42'
 snapshot=logs/f'controller_{checkpoint_step:04d}.json'
 state=json.loads(snapshot.read_text())
 assert state['last_step']==checkpoint_step and len(state['history'])==checkpoint_step
 assert state['c']>0
 # Collect/validate everything before moving files. All original bytes survive.
 future=[]
 for pattern in ['select_*.jsonl','raw_leaves_*.jsonl.gz','controller_*.json']:
  for p in logs.glob(pattern):
   match=re.search(r'_(\d+)\.',p.name)
   if match and int(match.group(1))>checkpoint_step:future.append(p)
 for pattern in ['rollouts_*.jsonl.gz','validation_*.json','validation_*.jsonl']:
  for p in formal.glob(pattern):
   match=re.search(r'_(\d+)\.',p.name)
   if match and int(match.group(1))>checkpoint_step:future.append(p)
 log_rewrites=[]
 for name in ['train_metrics.jsonl','branch_events.jsonl']:
  p=formal/name
  if p.exists():
   rows=p.read_text().splitlines();parsed=[json.loads(r) for r in rows]
   kept=[raw for raw,r in zip(rows,parsed) if r['step']<=checkpoint_step]
   if len(kept)!=len(rows):log_rewrites.append((p,kept))
 recovery=root/'recovery'/f'resume_step{checkpoint_step}_{time.time_ns()}';recovery.mkdir(parents=True)
 receipt=dict(checkpoint_step=checkpoint_step,controller_snapshot=str(snapshot),preserved=[])
 for p in future+[p for p,_ in log_rewrites]:
  dst=recovery/p.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256(p.read_bytes()).hexdigest();p.rename(dst)
  receipt['preserved'].append(dict(original=str(p),archive=str(dst),sha256=digest))
 for p,kept in log_rewrites:p.write_text(''.join(r+'\n' for r in kept))
 live=logs/'controller.json'
 if live.exists():live.rename(recovery/'controller_before.json')
 tmp=live.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(live)
 (recovery/'receipt.json').write_text(json.dumps(receipt,indent=2))
 return receipt
