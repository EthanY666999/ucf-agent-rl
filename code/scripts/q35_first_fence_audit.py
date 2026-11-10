"""CPU-only raw training trace audit. Never reads evaluation/test outcomes."""
import argparse, copy, gzip, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ucf')]
from first_fence import locate, segments

def audit(run, tokenizer, output):
    from cert_entropy import token_map, consequence_classes
    from tokens import fence_id
    from patr_core import make_env, distance, FrozenLakeAgent
    mapping,_=token_map(tokenizer); fence=fence_id(tokenizer)
    cfg=json.loads((run/'config.json').read_text())
    files=sorted(run.glob('rollouts_*.jsonl.gz')) or sorted(run.glob('rollouts.jsonl.gz'))
    assert files, 'No training trajectories'
    totals=Counter(); batches=[]; evidence={}; envs={}; examples=[]
    for file in files:
        with gzip.open(file,'rt') as f: rows=[json.loads(s) for s in f]
        # Only the actual four initial trajectories. Never pseudo/fallback rows.
        rows=[r for r in rows if str(r['lineage']) in ('0','1','2','3')]
        groups=defaultdict(list)
        for r in rows: groups[r['task']['uid']].append(r)
        assert all(sorted(str(r['lineage']) for r in rs)==['0','1','2','3'] for rs in groups.values())
        candidates=positive=zero=0; counts=Counter()
        for uid,rs in groups.items():
            if uid not in envs: envs[uid]=make_env(rs[0]['task'],cfg)
            for r in rs:
                env=copy.deepcopy(envs[uid]); initial_distance=distance(env)
                others=[x for x in rs if x['lineage']!=r['lineage']]
                ph=sum(x['terminal']=='success' for x in others)/3
                spans=list(segments(r['mask'])); assert len(spans)==len(r['trace'])
                counts['trajectories']+=1; covered=False
                for (start,end),tr in zip(spans,r['trace']):
                    counts['turns']+=1
                    truncated=tr.get('terminal')=='token_limit' and tr.get('action') is None
                    counts['length_truncated_turns']+=truncated
                    bd=locate(r['ids'],start,end,fence)
                    # Prefix-only capacity; exhausted nodes have no stochastic successor.
                    if bd is not None:
                        counts['turns_with_fence']+=1
                        used=sum(r['mask'][:bd]); rem=min(cfg['max_response_tokens']-used,cfg['max_context_tokens']-bd)
                        if rem>0:
                            counts['candidates']+=1; candidates+=1;covered=True
                            d=distance(env);den=initial_distance if getattr(env,'environment',None)=='sokoban' else env.nrow+env.ncol-2
                            S=0. if d is None else max(0.,min(1.,1-d/max(1,den)))
                            # Softmax gives all tokens positive mass; >=2 consequence
                            # classes imply positive H_cons. Only positivity is needed
                            # to compute calibrated aggregate expectation, not each rho.
                            h_positive=len(set(consequence_classes(env).values()))>1
                            pos=(1-ph)*S>0 or (ph>0 and h_positive)
                            positive+=pos;zero+=not pos
                        else: counts['prefix_budget_exhausted']+=1
                        a=mapping.get(r['ids'][bd],0) if bd<end else 0
                        parsed=int(FrozenLakeAgent._parse_model_response(None,tr['response'])[1])
                        counts['first_fence_parse_compared']+=1
                        counts['first_fence_parse_mismatch']+=a!=parsed
                        counts['missing_post_fence_token']+=bd==end
                        if a!=parsed and len(examples)<30:
                            examples.append(dict(file=file.name,uid=uid,lineage=r['lineage'],turn=tr['turn'],first_action=a,parsed_action=parsed,prefix_end=bd))
                    # Disjoint historical v2 exclusion categories: truncation first.
                    if truncated: counts['v2_length_excluded']+=1
                    elif bd!=start+1 or mapping.get(r['ids'][start+1] if start+1<end else -1,0)!=tr.get('action'):
                        counts['v2_format_excluded_nontruncated']+=1
                    else: counts['v2_format_eligible_nontruncated']+=1
                    if tr.get('action') is not None: env.step(tr['action'])
                counts['trajectories_with_candidate']+=covered
        n=len(groups);floor=.03*candidates/n;capacity=(positive+.03*zero)/n
        assert floor<=2, 'Unchanged rho_min exceeds target; manual review'
        batches.append(dict(step=int(file.stem.split('_')[-1].split('.')[0]) if '_' in file.stem else 0,tasks=n,**counts,
                            expected_forks=min(2.,capacity),reachable_max=capacity,rho_floor=floor))
        totals.update(counts)
        evidence[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        if len(batches)%50==0: print(json.dumps({'audited_batches':len(batches)}),flush=True)
    ratio=totals['first_fence_parse_mismatch']/max(1,totals['first_fence_parse_compared'])
    report=dict(version='UCF-v2.1',source=str(run),training_only=True,counts=dict(totals),
      fence_turn_fraction=totals['turns_with_fence']/totals['turns'],
      candidate_turn_coverage=totals['candidates']/totals['turns'],
      candidate_trajectory_coverage=totals['trajectories_with_candidate']/totals['trajectories'],
      expected_forks_per_task=sum(b['expected_forks']*b['tasks'] for b in batches)/sum(b['tasks'] for b in batches),
      expectation_method='Unchanged calibrated target min(2, reachable support); positive finite softmax over all action variants. No per-node rho or entropy inferred from top-1 logs. Numerical GPU coverage still pending.',
      mismatch_fraction=ratio,mismatch_threshold=.02,passed=ratio<=.02,
      batches=batches,mismatch_examples=examples,record_sha256=evidence)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    r=audit(a.run,AutoTokenizer.from_pretrained('/data/models/Qwen3.5-2B'),a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ['batches','record_sha256','mismatch_examples']}))
