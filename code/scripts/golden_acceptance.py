"""Technical GRPO acceptance before launching PaTR; no test-set tuning."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/formal/grpo_seed42'
s=json.loads((run/'status.json').read_text())
cfg=json.loads((run/'config.json').read_text())
rows=[json.loads(x) for x in (run/'train_metrics.jsonl').read_text().splitlines()]
assert s['status']=='complete' and s['step']==cfg['steps']==400
assert len(rows)==400 and [r['step'] for r in rows]==list(range(1,401))
for row in rows:
    assert all(math.isfinite(row[k]) for k in ['loss','grad_norm','kl','max_parameter_change'])
assert any(row['max_parameter_change']>0 for row in rows), 'No model updates'
final=json.loads((run/'final_test.json').read_text())
assert final['n_tasks']==100 and final['n_trajectories']==400
assert final['success_rate_pct']>0, 'Zero success requires investigation before PaTR'
assert (run/'final_model/config.json').exists()
report=dict(status='passed',meaning='Technical training/evaluation acceptance, not scientific superiority',
            final=final,sft_test=json.loads((ROOT/'runs/golden_sft/sft_test.json').read_text()))
(run/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
