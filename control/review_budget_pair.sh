#!/usr/bin/env bash
# Postprocess two verified boundaries on CPU. Entirely separate from training.
set -euo pipefail
state=${1:?state directory}
step=${2:?completed paired boundary}
[[ $step =~ ^[1-9][0-9]*$ ]]
cd "$state"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/data/venvs/qwen35/bin/python
tag=$(printf '%04d' "$step")
out=$state/reports
mapfile -t dirs < <("$py" - "$step" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path('plan.json').read_bytes())
items={e['method']: Path(e['output']) for e in p['entries'] if e['stop_step']==int(sys.argv[1])}
assert items.keys()=={'ucf','grpo'}
print(Path(p['manager_script']).parent/'review_pilot.py')
print(items['ucf']); print(items['grpo'])
PY
)
[[ ${#dirs[@]} == 3 ]]
review=${dirs[0]}; ucf=${dirs[1]}; grpo=${dirs[2]}
for run in "$ucf" "$grpo"; do
    test -s "$run/stage_$tag.json"
done
"$py" "$review" --run "$ucf" --comparison "$grpo" --step "$step" --output "$out/paired_review_$tag.json"
"$py" "$review" --run "$grpo" --step "$step" --output "$out/grpo_review_$tag.json"
"$py" paired_budget_accounting.py --ucf "$ucf" --grpo "$grpo" --step "$step" --output "$out/paired_accounting_$tag.json"
for arm in ucf grpo; do
    if [[ $arm == ucf ]]; then run=$ucf; else run=$grpo; fi
    "$py" audit_budget_updates.py --run "$run" --through "$step" --output "$out/${arm}_updates_$tag.json"
    "$py" export_budget_boundary.py --run "$run" --step "$step" --output "$out/${arm}_boundary_$tag.tar.gz"
done
"$py" - "$out" "$tag" <<'PY'
import hashlib,json,sys
from pathlib import Path
r,tag=Path(sys.argv[1]),sys.argv[2]
acct=json.loads((r/f'paired_accounting_{tag}.json').read_bytes())
review=json.loads((r/f'paired_review_{tag}.json').read_bytes())
paired=review['paired_comparison']['overall']
for a,b in [('success_pp','success'),('pass_at_4_pp','pass4')]:
    x=acct['difference'][a]; y=paired[b]
    assert abs(x['estimate']-y['difference_percentage_points'])<1e-9
    assert all(abs(l-r)<1e-8 for l,r in zip(x['task_bootstrap_95'],y['percentile_95_ci_percentage_points']))
paths=[r/f'{stem}_{tag}.{ext}' for stem,ext in (
    ('paired_review','json'),('grpo_review','json'),('paired_accounting','json'),
    ('ucf_updates','json'),('grpo_updates','json'),('ucf_boundary','tar.gz'),('grpo_boundary','tar.gz'))]
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
target=r/f'paired_export_{tag}.json'; tmp=target.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(step=int(tag),passed=True,final_test_used=False,files=files),indent=2)+'\n')
tmp.replace(target)
print(json.dumps({'step':int(tag),'difference':acct['difference'],'ucf_over_grpo':acct['ucf_over_grpo']}))
PY
