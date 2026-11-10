"""CPU equivalence on actual sampled IDs plus complete vocabulary screening."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

old=Path('/home/vipuser/decision-point-ucf-v3-scoped-20261110/code')
candidate=Path('/home/vipuser/ucf-prefix-cpu-candidate-20261110')
sys.path[:0]=[str(candidate),str(candidate/'ucf'),str(old),str(old/'ucf')]
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
before_f=load('before_f',old/'ucf/first_fence.py')
before_p=load('before_p',old/'response_protocol.py')
after_f=load('after_f',candidate/'ucf/first_fence.py')
after_p=load('after_p',candidate/'response_protocol.py')
from token_boundaries import fence_token_positions
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B',local_files_only=True)
vocab=tok.get_vocab();ids=list(vocab.values())
pieces=tok.backend_tokenizer.decode_batch([[i] for i in ids],skip_special_tokens=True)
candidate_ids={i for piece,i in vocab.items() if '`' in piece}
assert all('`' not in text or i in candidate_ids for i,text in zip(ids,pieces))
del pieces
cases=[]
for tag in ['S','F']:
    p=Path('/data/runs/ucf_v3_numerical_scoped_20261110')/tag/'ucf_logs/raw_leaves_0001.jsonl.gz'
    with gzip.open(p,'rt') as f:
        for line in f:
            b=json.loads(line);start=None
            for j,m in enumerate(b['mask']+[0]):
                if m and start is None:start=j
                if not m and start is not None:
                    cases.append((f'{tag}:{b["kind"]}:{b["task_id"]}:{b["lineage"]}:{start}',b['ids'][start:j]));start=None
for text in ['prefix ```Right``` suffix','中文🙂 ```\nLeft\n``` tail',
             'draft ``` Right </think> finally ```Up```','no block at all',
             'word '*100+'```Down```', '``','```','````','```1```','``` Right```中文']:
    cases.append(('constructed',tok.encode(text,add_special_tokens=False)))
# Fragment a fence and insert skipped special tokens: no retokenization of the
# actual recorded cases above, and the candidate cannot read a later suffix.
tick=tok.encode('`',add_special_tokens=False)
cases.append(('fragmented_skipped_special',tick+[tok.eos_token_id]+tick+tick+tok.encode('Right```',add_special_tokens=False)))
timings={k:0. for k in ['old_fence','new_fence','old_cutoff','new_cutoff']}
checked=[]
for label,seq in cases:
    results=[]
    for key,fn in [('old_fence',before_f.locate_token_prefix),('new_fence',after_f.locate_token_prefix)]:
        t=time.monotonic();results.append(fn(seq,0,len(seq),tok));timings[key]+=time.monotonic()-t
    assert results[0]==results[1],(label,results)
    boundary=results[0]
    if boundary is not None:
        class NoSuffix(list):
            def __getitem__(self,index):
                if isinstance(index,slice):assert (index.stop or len(self))<=boundary
                else:assert index<boundary
                return super().__getitem__(index)
        assert after_f.locate_token_prefix(NoSuffix(seq),0,len(seq),tok)==boundary
    out=SimpleNamespace(token_ids=seq,logprobs=list(range(len(seq))),text=tok.decode(seq,skip_special_tokens=True),finish_reason='length')
    branch=SimpleNamespace(ids=[],mask=[])
    r=[]
    for key,fn in [('old_cutoff',before_p.truncate_response),('new_cutoff',after_p.truncate_response)]:
        t=time.monotonic();value,meta=fn(branch,out,tok);timings[key]+=time.monotonic()-t
        r.append((value.token_ids,value.logprobs,value.text,value.finish_reason,meta))
    assert r[0]==r[1],label
    checked.append(dict(label=label,tokens=len(seq),boundary=boundary,sha256=hashlib.sha256(repr(seq).encode()).hexdigest()))
    if len(checked)%10==0:print(json.dumps(dict(checked=len(checked),total=len(cases),timings=timings)),flush=True)
report=dict(passed=True,actual_sampled_ids_retained=True,checked=len(checked),vocabulary_size=len(vocab),
            candidate_token_ids=len(candidate_ids),all_single_token_backticks_covered=True,
            prefix_causality_no_suffix_access=True,timings_seconds=timings,cases=checked,
            candidate_sha256={str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [candidate/'token_boundaries.py',candidate/'response_protocol.py',candidate/'ucf/first_fence.py']})
(candidate/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
