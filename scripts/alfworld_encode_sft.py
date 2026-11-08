#!/usr/bin/env python3
"""CPU-only SFT mask/token-budget audit. Does not load or train model weights."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from alfworld_adapter.sft_data import encode_example

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=16384)
    a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    count=total=loss=maximum=0;episodes=set();digest=hashlib.sha256()
    with gzip.open(a.input,'rt') as f:
        for line in f:
            item=json.loads(line)
            if item['split']!='train':raise ValueError('Non-training split in SFT input')
            encoded=encode_example(tok,item['messages'],a.max_tokens)
            assert all(x==-100 for x in encoded['labels'][:encoded['response_start']])
            assert encoded['labels'][encoded['response_start']:]==encoded['input_ids'][encoded['response_start']:]
            count+=1;total+=len(encoded['input_ids']);loss+=encoded['loss_tokens']
            maximum=max(maximum,len(encoded['input_ids']));episodes.add(item['episode_id'])
            digest.update(json.dumps(encoded,separators=(',',':')).encode())
    report=dict(status='passed',gpu_used=False,model_loaded=False,examples=count,episodes=len(episodes),
                total_input_tokens=total,supervised_tokens=loss,max_example_tokens=maximum,
                encoded_sha256=digest.hexdigest(),enable_thinking=False,
                tokenizer_sha256=hashlib.sha256((a.tokenizer/'tokenizer.json').read_bytes()).hexdigest(),
                chat_template_sha256=hashlib.sha256((a.tokenizer/'chat_template.jinja').read_bytes()).hexdigest()
                    if (a.tokenizer/'chat_template.jinja').exists() else None,
                input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest())
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
