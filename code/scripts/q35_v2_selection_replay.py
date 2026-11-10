"""Replay serialized v2 selection inputs; compare output bytes, without sampling."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
import ucf_select

def replay(run,output):
    cfg=json.loads((run/'config.json').read_text());limit=json.loads((run/'status.json').read_text())['step']
    rng=np.random.default_rng(cfg['seed']);checked=[]
    for step in range(1,limit+1):
        seed=int(rng.integers(0,2**31-1));path=run/'ucf_logs'/f'select_{step:04d}.jsonl'
        original=path.read_bytes();rows=[json.loads(x) for x in original.splitlines()]
        tasks={r['task_id']:{'uid':r['uid']} for r in rows};cs={i:[] for i in tasks}
        for r in rows:
            cs[r['task_id']].append(dict(branch=NS(task_id=r['task_id']),j=r['j'],turn=r['turn'],
              stratum={'parse_mismatch':r['parse_mismatch']},recorded=(r['w'],r['p_hat_minus_j'],r['baseline'])))
        with tempfile.TemporaryDirectory() as td,patch.object(ucf_select,'feature',side_effect=lambda n,*a:n['recorded']):
            dest=Path(td)/'select.jsonl'
            ucf_select.select_all(cs,[],tasks,rows[0]['c_step'],seed,step,dest,cfg['ucf_variant'])
            assert dest.read_bytes()==original, f'selection differs at step {step}'
        checked.append(dict(step=step,rows=len(rows),sha256=hashlib.sha256(original).hexdigest()))
    result=dict(passed=True,through_step=limit,steps=checked,
        scope='byte-identical replay of recorded selection inputs (w, leave-one-out statistics, c_step); rho, RNG uniform, decision and JSON bytes recomputed. Source feature/probe code separately byte-identical to v2.',GPU_used=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=replay(a.run,a.output);print(json.dumps({k:v for k,v in r.items() if k!='steps'}))
