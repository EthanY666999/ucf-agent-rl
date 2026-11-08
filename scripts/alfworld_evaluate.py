#!/usr/bin/env python3
"""Standalone vLLM ALFWorld evaluation; does NOT use the four-action trainer.

No GPU execution occurs unless --execute-gpu is explicitly supplied. This
entry point is syntax/CPU tested only until a collaborator provides GPU smoke.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from alfworld_adapter import ALFWorldEpisode, parse_action
from alfworld_adapter.sft_data import token_ids

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--games-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=['valid_seen','valid_unseen'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-response-tokens',type=int,default=4096)
    p.add_argument('--max-model-len',type=int,default=16384)
    p.add_argument('--temperature',type=float,default=.4)
    p.add_argument('--tensor-parallel-size',type=int,default=1)
    p.add_argument('--gpu-memory-utilization',type=float,default=.8)
    p.add_argument('--limit',type=int,default=0,help='0 = complete split; positive = smoke subset')
    p.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args()
    if not a.execute_gpu:
        p.error('GPU execution disabled; add --execute-gpu on the collaborator machine only')
    if not a.model.is_dir() or not list(a.model.glob('*.safetensors')):
        p.error('Provide a local model with weights; this script never downloads weights')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be new/empty')
    from vllm import LLM, SamplingParams
    engine=LLM(model=str(a.model),dtype='bfloat16',generation_config='vllm',
               max_model_len=a.max_model_len,tensor_parallel_size=a.tensor_parallel_size,
               gpu_memory_utilization=a.gpu_memory_utilization)
    tokenizer=engine.get_tokenizer()
    manifest=json.loads(a.manifest.read_text());games=manifest['splits'][a.split]
    if a.limit:games=games[:a.limit]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for game in games:
        env=ALFWorldEpisode(a.games_root/game,a.seed)
        turns=[]
        try:
            while not env.done:
                messages=[{'role':'user','content':env.prompt()}]
                prefix=token_ids(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                      enable_thinking=False))
                budget=min(a.max_response_tokens,a.max_model_len-len(prefix))
                if budget<=0:
                    turns.append({'terminal':'context_limit'});break
                stable=int.from_bytes(hashlib.sha256(f'{a.seed}:{game}:{len(turns)}'.encode()).digest()[:4],'big')
                out=engine.generate([{'prompt_token_ids':list(prefix)}],
                     SamplingParams(temperature=a.temperature,top_p=1,top_k=-1,max_tokens=budget,
                                    seed=stable,logprobs=1),use_tqdm=False)[0].outputs[0]
                record=dict(token_ids=list(out.token_ids),response=out.text,finish_reason=out.finish_reason,
                            old_logprobs=[float(lp[t].logprob) for t,lp in zip(out.token_ids,out.logprobs)])
                if out.finish_reason=='length':
                    record['terminal']='token_limit';turns.append(record);break
                action=parse_action(out.text)
                record['parsed_action']=action
                record['valid_action']=action in env.admissible
                # Invalid output is an environment action attempt, not a resample.
                obs,reward,done=env.step(action if action is not None else '__invalid_action__')
                record.update(next_observation=obs,reward=reward,done=done)
                turns.append(record)
            row=dict(game=game,split=a.split,won=env.won,turns=turns)
            rows.append(row)
            with (a.output_dir/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        finally:env.close()
    metrics=dict(split=a.split,episodes=len(rows),successes=sum(x['won'] for x in rows),
                 success_rate=sum(x['won'] for x in rows)/len(rows),
                 full_split=a.limit==0,max_steps=50,history_length=2,temperature=a.temperature,
                 truncated_responses=sum(t.get('terminal')=='token_limit' for x in rows for t in x['turns']),
                 protocol='first fenced action; differs from published XML prompt; Qwen3.5 is a different backbone')
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (a.output_dir/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
