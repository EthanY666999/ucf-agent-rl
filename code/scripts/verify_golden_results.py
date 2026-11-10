"""Verify published evaluation numbers and branch-pair statistics, CPU/stdlib only."""
import collections,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/golden-sft-20261031';AUDIT=ROOT/'experiments/golden-controls-20261031/results'
def read(ps):return [json.loads(s) for p in ps for s in p.read_text().splitlines()]
def scores(rs):
 g=collections.defaultdict(list)
 for r in rs:g[r['task']['uid']].append(r)
 assert all(len(v)==4 for v in g.values())
 return 100*sum(r['reward']>0 for r in rs)/len(rs),100*sum(any(r['reward']>0 for r in v) for v in g.values())/len(g)
report=json.loads((AUDIT/'overlap_report.json').read_text());exclude=set(report['excluded_uids']);assert len(exclude)==91
paths={'old_patr':[ROOT/f'results/supplemental500/patr/batch{i}.jsonl' for i in range(1,6)],'sft':[RESULT/f'completed400/runs/golden_sft/supplemental500/sft/batch{i}.jsonl' for i in range(1,6)],'sft_grpo400':[RESULT/f'completed400/runs/golden_sft/supplemental500/grpo/batch{i}.jsonl' for i in range(1,6)],'sft_patr400':[RESULT/f'completed400/runs/golden_sft/supplemental500/patr/batch{i}.jsonl' for i in range(1,6)],'sft_patr500':[RESULT/f'completed500/runs/finish500/supplemental500/batch{i}.jsonl' for i in range(1,6)]}
for model,files in paths.items():
 rs=read(files);assert len(rs)==2000
 for group,rr in [('original500',rs),('unseen409',[r for r in rs if r['task']['uid'] not in exclude]),('seen91',[r for r in rs if r['task']['uid'] in exclude])]:
  a,b=scores(rr);expected=report['models'][model][group]
  assert abs(a-expected['success_pct'])<1e-10 and abs(b-expected['pass4_pct'])<1e-10,(model,group)
  eligible=[r for r in rr if r['trace'] and all(t['action'] in (1,2,3,4) for t in r['trace'])]
  assert len(eligible)==expected['zero_invalid_n']
for label in ['patr400','patr401_500']:
 ps=read([AUDIT/f'paired_parents_{label}.jsonl']);parents=read([AUDIT/f'parent_outcomes_{label}.jsonl']);bykey={(r['step'],r['uid'],r['parent']):r for r in parents};assert len(bykey)==len(parents)
 r=json.loads((AUDIT/f'branch_report_{label}.json').read_text());assert len(ps)==r['counts']['different_selected_pair_events']
 for key in ['mixed','success']:
  observed=sum(p[key+'_delta'] for p in ps)/len(ps);expected=r['paired_high_minus_low_'+('mixed' if key=='mixed' else 'child_success')]['mean'];assert abs(observed-expected)<1e-12
manifest=json.loads((ROOT/'GOLDEN_SYNC_MANIFEST.json').read_text())
for name,entry in manifest['files'].items():
 b=(ROOT/name).read_bytes();assert hashlib.sha256(b).hexdigest()==entry['sha256'] and len(b)==entry['bytes'],name
print('PASS: five models original500 / filtered409 / seen91, legal-only denominators, two branch audits and complete sync manifest.')
