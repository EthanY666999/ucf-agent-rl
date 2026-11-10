"""v2.2 audit: change only trajectory j's suffix, candidates and final reward."""
import argparse,copy,hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'ucf')]
from ucf_select import calibrate,feature
from ucf_corrected import leave_question_out_constants,select_corrected

def audit():
 first=[NS(task_id=k,lineage=str(j),reward=float(j%2),terminal='success' if j%2 else 'hole') for k in range(64) for j in range(4)]
 candidates={k:[dict(branch=NS(task_id=k,ids=[100,k,j]),j=j,turn=1,S=.3,H_cons=.7) for j in range(4)] for k in range(64)}
 cases=[]
 for change in [False,True]:
  leaves=copy.deepcopy(first);cs=copy.deepcopy(candidates)
  if change:
   leaves[0].reward=1.;leaves[0].terminal='success'
   cs[0].append(dict(branch=NS(task_id=0,ids=[100,0,0,999]),j=0,turn=2,S=.8,H_cons=.9))
  weights={k:[feature(n,leaves)[0] for n in ns] for k,ns in cs.items()}
  constants=leave_question_out_constants(weights)
  with tempfile.TemporaryDirectory() as td:
   _,rows=select_corrected(cs,leaves,[{'uid':str(k)} for k in range(64)],constants,42,1,Path(td)/'select.jsonl')
  cases.append(dict(c_k=constants[0],j0_prefix=rows[0],other_three=[r for r in rows if r['task_id']==0 and r['turn']==1 and r['j']!=0],legacy_c=calibrate(list(weights.values()))))
 a,b=cases
 checks=dict(c_k_unchanged=a['c_k']==b['c_k'],current_j_prefix_row_unchanged=a['j0_prefix']==b['j0_prefix'],current_j_ph_baseline_unchanged=all(a['j0_prefix'][k]==b['j0_prefix'][k] for k in ['w','p_hat_minus_j','baseline']),other_three_rho_may_change=[x['rho'] for x in a['other_three']]!=[x['rho'] for x in b['other_three']])
 passed=all(checks[k] for k in ['c_k_unchanged','current_j_prefix_row_unchanged','current_j_ph_baseline_unchanged'])
 return dict(passed=passed,requirement='All other questions and own other three trajectories fixed. Only trajectory j suffix, later candidates and final reward change. c_k and unchanged-prefix rho on j must remain equal. Other three rho need not remain equal.',checks=checks,cases=cases,S05_legacy_affected=True,S05_modified=False,GPU_used=False,source_sha256={f:hashlib.sha256((ROOT/'ucf'/f).read_bytes()).hexdigest() for f in ['ucf_select.py','ucf_corrected.py']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
