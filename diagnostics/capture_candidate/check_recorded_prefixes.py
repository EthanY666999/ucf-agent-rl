"""CPU replay of observer boundaries on immutable sampled token histories."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from first_fence import locate_token_prefix, segments
from response_protocol import first_block
from original_logits import CaptureSession, OriginalLogitsObserver, request_metadata

assert not torch.cuda.is_available(), 'This audit must not use the active GPU'
root=Path('/home/vipuser/decision-point-ucf-v3-capture-candidate-20261110')
run=Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42')
tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
cases=[];started=time.monotonic()
for step in [1,7,9]:
    source=run/f'ucf_logs/raw_leaves_{step:04d}.jsonl.gz'
    count=0
    with gzip.open(source,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['kind']!='initial':continue
            for turn,(start,end) in enumerate(segments(row['mask']),1):
                output=row['ids'][start:end]
                bd=locate_token_prefix(output,0,len(output),tok)
                closed=bd is not None and first_block(tok.decode(output[:bd],skip_special_tokens=True)) is not None
                expected=bd is not None and bd<len(output) and not closed
                live=[];obs=OriginalLogitsObserver(None,'cpu',False)
                # Fake CPU logits suffice to test placement/request identity;
                # probability correctness is reserved for real GPU admission.
                with CaptureSession(tok,1,16) as cap:
                    p=SamplingParams(temperature=1.,top_p=1.,top_k=-1,seed=42,
                        extra_args=request_metadata(cap.serial,row['task_id'],row['lineage'],turn-1))
                    obs.update_state(BatchUpdate(1,[],[(0,p,row['ids'][:start],live)],[]))
                    logits=torch.zeros(1,16)
                    for token in output:
                        assert obs.apply(logits) is logits
                        live.append(token)
                assert bool(cap.records)==expected,(step,row['task_id'],row['lineage'],turn,bd)
                if expected:
                    prefix,_=next(iter(cap.records.values()))
                    assert prefix==tuple(row['ids'][:start+bd])
                cases.append(dict(step=step,task_id=row['task_id'],lineage=row['lineage'],
                                  turn=turn,response_tokens=len(output),boundary=bd,
                                  candidate=expected,passed=True))
                count+=1
                if count>=24:break
            if count>=24:break
result=dict(passed=True,cpu_only=True,cases=cases,seconds=time.monotonic()-started,
            observer_sha256=hashlib.sha256((root/'code/ucf/original_logits.py').read_bytes()).hexdigest(),
            interpretation='Recorded token boundary equality only; GPU sampling admission still required')
(root/'recorded_prefix_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,cases=len(cases),candidates=sum(x['candidate'] for x in cases),
                      seconds=result['seconds'],cpu_only=True)))
