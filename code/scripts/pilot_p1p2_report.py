"""Read-only P1/P2 reporting and preregistered 150-step screening.

No torch, model, GPU, queue mutation, or final/test-set evaluation. Writes only
explicit report paths. Historical raw records stay on the server.
"""
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

GROUPS = ('all_failure', 'mixed', 'all_success')
COUNTS = ('task_count', 'rescued_tasks', 'candidate_count', 'sum_rho',
          'selected_count', 'dead_sum_rho', 'dead_selected_count', 'early_count',
          'early_sum_rho', 'early_selected_count')
EVAL_STEPS = (100, 125, 150)


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_metrics(run, upto=150):
    rows = {}
    path = Path(run) / 'train_metrics.jsonl'
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            # A writer may be in the middle of its last append. Do not count it.
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            step = int(row['step'])
            if step > upto:
                break
            if step in rows:
                raise ValueError(f'duplicate committed metric step {step}')
            rows[step] = row
    return rows


def clean_number(value):
    return float(value) if finite(value) else None


def compact_step(row, selection):
    b = row.get('ucf_preupdate_fullbatch_components', {}).get('B', {})
    values = {'kl': row.get('kl'), 'part_b_grad_norm': b.get('grad_norm'),
              'step_seconds': row.get('step_seconds')}
    return dict(step=int(row['step']), selection=selection,
                group_counts={k: selection[k]['task_count'] for k in GROUPS},
                **{k: clean_number(v) for k, v in values.items()},
                invalid_metrics={k: repr(v) for k, v in values.items() if not finite(v)},
                mean_forks=clean_number(row.get('mean_forks')),
                reachable_expected_forks=row.get('pilot_solvability', {}).get('calibration'),
                original_controller=row.get('c_step'))


def read_validation(run, step):
    # Deliberately never open final_test.json, supplemental500, or test traces.
    path = Path(run) / f'validation_{step:04d}.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get('split') != 'validation':
        raise ValueError(f'Expected fixed validation split: {path}')
    for key in ('success_rate_pct', 'pass_at_4_pct'):
        if not finite(d.get(key)) or not 0 <= d[key] <= 100:
            raise ValueError(f'invalid validation metric {key}: {path}')
    return dict(step=step, success_rate=d['success_rate_pct'] / 100.,
                pass_at_4=d['pass_at_4_pct'] / 100., n_tasks=d.get('n_tasks'),
                n_trajectories=d.get('n_trajectories'), sha256=sha256(path))


def empty_counts():
    return {k: 0 for k in COUNTS}


def selection_from_live(row):
    stats = row.get('pilot_solvability')
    if not stats:
        return None
    source = stats.get('per_group', stats)
    out = {}
    for group in GROUPS:
        raw = source.get(group)
        if raw is None or any(k not in raw for k in COUNTS):
            return None
        vals = {k: raw[k] for k in COUNTS}
        if any(not finite(v) or v < 0 for v in vals.values()):
            raise ValueError('Invalid selection counts')
        if vals['rescued_tasks'] > vals['task_count'] or vals['dead_selected_count'] > vals['selected_count']:
            raise ValueError('Inconsistent selection counts')
        out[group] = vals
    return out


def aggregate(rows, start=1, end=150):
    wanted = list(range(start, end + 1))
    chosen = [r for r in rows if start <= r['step'] <= end]
    missing = sorted(set(wanted) - {r['step'] for r in chosen})
    by_group = {}
    for group in GROUPS:
        s = {key: sum(r['selection'][group][key] for r in chosen) for key in COUNTS}
        by_group[group] = dict(s,
            dead_actual_budget_share=ratio(s['dead_selected_count'], s['selected_count']),
            dead_expected_budget_share=ratio(s['dead_sum_rho'], s['sum_rho']),
            rescue_rate=ratio(s['rescued_tasks'], s['task_count']),
            early_expected_budget_share=ratio(s['early_sum_rho'], s['sum_rho']),
            early_actual_budget_share=ratio(s['early_selected_count'], s['selected_count']),
            early_mean_rho=ratio(s['early_sum_rho'], s['early_count']))
    numeric = {}
    for name in ('kl', 'part_b_grad_norm', 'step_seconds'):
        vals = [r[name] for r in chosen if finite(r.get(name))]
        numeric[name] = dict(count=len(vals), mean=sum(vals)/len(vals) if vals else None,
                             max=max(vals) if vals else None, p99=quantile(vals, .99))
    return dict(start=start, end=end, observed_steps=len(chosen), missing_steps=missing,
                per_group=by_group, diagnostics=numeric,
                invalid_metrics=[dict(step=r['step'], fields=r['invalid_metrics'])
                                 for r in chosen if r.get('invalid_metrics')])


def build_reference(run, audit_candidates, upto=150):
    """Use prior BFS-aligned audit file; raw trajectories are not copied/read."""
    run, audit_candidates = Path(run), Path(audit_candidates)
    metrics = read_metrics(run, upto)
    if set(metrics) != set(range(1, upto + 1)):
        raise ValueError('S05 reference metrics are incomplete')
    stats = {s: {g: empty_counts() for g in GROUPS} for s in metrics}
    task_ids = defaultdict(set)
    rescued_ids = defaultdict(set)
    with gzip.open(audit_candidates, 'rt') as stream:
        for line in stream:
            n = json.loads(line)
            step = n['step']
            if step > upto:
                break
            if step not in stats:
                continue
            group = n['group_type']; s = stats[step][group]
            task_ids[(step, group)].add(n['task_id'])
            s['candidate_count'] += 1; s['sum_rho'] += n['rho']
            s['selected_count'] += bool(n['selected'])
            dead = n['state_class'] in ('structural_deadlock', 'budget_exhausted')
            if dead:
                s['dead_sum_rho'] += n['rho']; s['dead_selected_count'] += bool(n['selected'])
            if n['turn'] <= 3:
                s['early_count'] += 1; s['early_sum_rho'] += n['rho']
                s['early_selected_count'] += bool(n['selected'])
            if n['R_cf'] is not None and n['R_cf'] > 0:
                rescued_ids[(step, group)].add(n['task_id'])
    compact = []
    for step, row in sorted(metrics.items()):
        for g in GROUPS:
            s = stats[step][g]
            # Initial task count includes all tasks, not only selected tasks.
            s['task_count'] = row['initial_group_counts'].get(g, 0)
            if len(task_ids[(step, g)]) != s['task_count']:
                raise ValueError('Reference candidate/task grouping is incomplete')
            s['rescued_tasks'] = len(rescued_ids[(step, g)])
        compact.append(compact_step(row, stats[step]))
    summary = aggregate(compact, 1, upto)
    if summary['invalid_metrics']:
        raise ValueError('Reference contains invalid training diagnostics')
    thresholds = {key: 3 * summary['diagnostics'][key]['p99']
                  for key in ('kl', 'part_b_grad_norm')}
    validation = {str(s): read_validation(run, s) for s in range(25, upto + 1, 25)}
    return dict(schema_version=1, kind='S05_historical_v2_reference', cutoff=upto,
                source_run=str(run), rows=compact, cumulative=summary,
                validation=validation, spike_thresholds=thresholds,
                spike_rule='Any nonfinite/missing diagnostic is ineligible; any step above 3x linear-interpolated S05 steps1-150 P99 is a spike.',
                provenance={str(run/'train_metrics.jsonl'): sha256(run/'train_metrics.jsonl'),
                            str(audit_candidates): sha256(audit_candidates),
                            str(run/'config.json'): sha256(run/'config.json')},
                definitions=definitions())


def definitions():
    return dict(a='All-failure initial groups: selected dead candidates / all selected candidates. Expected sum-rho share also reported. Dead=structural OR minimum distance>remaining turns.',
                b='Number of initially all-failure TASKS with >=1 actually executed successful counterfactual / ALL initially all-failure tasks (including tasks with zero forks).',
                c='Early=turn1-3. Early mean rho includes gated rho0 candidates; expected budget=sum early rho/sum all rho within group.',
                d='Groups use original four trajectories before counterfactual branches.',
                e='Fixed validation only; comparison uses arithmetic mean at steps100,125,150.',
                f='Part B gradient is ucf_preupdate_fullbatch_components.B.grad_norm, before optimizer minibatches/clipping; KL is logged training KL; seconds exclude validation as in S05.',
                comparison_limit='S05 is v2; pilots are v2.2 plus stated changes. Historical comparison is a screening reference, not a gate-only causal ablation.',
                reward_gradient_limit='No claim that gating preserves the entire PPO+KL update; KL and token normalization are retained by protocol.')


def evaluate_150(rows, validation, reference):
    stats = aggregate(rows, 1, 150)
    missing = list(stats['missing_steps'])
    missing_eval = [s for s in EVAL_STEPS if not validation.get(str(s))]
    base_eval = [reference['validation'].get(str(s)) for s in EVAL_STEPS]
    if any(x is None for x in base_eval):
        raise ValueError('Reference final-three validation points missing')
    baseline_mean = sum(x['success_rate'] for x in base_eval) / 3
    pilot_mean = None if missing_eval else sum(validation[str(s)]['success_rate'] for s in EVAL_STEPS)/3
    af = stats['per_group']['all_failure']; ref_af = reference['cumulative']['per_group']['all_failure']
    checks = dict(a=af['dead_actual_budget_share'] is not None and af['dead_actual_budget_share'] < .10,
                  b=af['rescue_rate'] is not None and ref_af['rescue_rate'] is not None and af['rescue_rate'] > ref_af['rescue_rate'],
                  e=pilot_mean is not None and pilot_mean >= baseline_mean - .03)
    spikes = []
    for row in rows:
        if 1 <= row['step'] <= 150:
            for name, limit in reference['spike_thresholds'].items():
                if finite(row.get(name)) and row[name] > limit:
                    spikes.append(dict(step=row['step'], metric=name, value=row[name], threshold=limit))
    checks['f'] = not stats['invalid_metrics'] and not spikes and all(
        stats['diagnostics'][name]['count'] == 150 for name in ('kl', 'part_b_grad_norm'))
    pending = bool(missing or missing_eval)
    return dict(status='pending' if pending else 'pass' if all(checks.values()) else 'fail',
                eligible_for_400=(not pending and all(checks.values())), checks=checks,
                missing_steps=missing, missing_validation_steps=missing_eval,
                all_failure=af, baseline_all_failure=ref_af,
                validation_last3_mean=pilot_mean, baseline_validation_last3_mean=baseline_mean,
                validation_threshold=baseline_mean-.03, spikes=spikes,
                invalid_metrics=stats['invalid_metrics'], spike_thresholds=reference['spike_thresholds'])


def report_run(run, reference, upto=150):
    run=Path(run); metrics=read_metrics(run, upto); compact=[]; missing_selection=[]
    for step, row in sorted(metrics.items()):
        selection=selection_from_live(row)
        if selection is None:
            missing_selection.append(step)
            continue
        compact.append(compact_step(row, selection))
    observed=max(metrics, default=0)
    validations={str(s):read_validation(run,s) for s in range(25,min(upto,observed)+1,25)}
    windows=[]
    for end in range(25,min(upto,observed)+1,25):
        start=end-24
        windows.append(dict(start=start,end=end,pilot=aggregate(compact,start,end),
                            S05=aggregate(reference['rows'],start,end),
                            validation=validations.get(str(end)),
                            S05_validation=reference['validation'].get(str(end))))
    return dict(schema_version=1, kind='P1_P2_short_validation', source_run=str(run),
                last_committed_metric_step=observed, missing_selection_steps=missing_selection,
                rows=compact, windows=windows, validation=validations,
                cumulative=aggregate(compact,1,min(observed,upto)) if observed else None,
                decision_150=evaluate_150(compact,validations,reference), definitions=definitions())


def choose_upgrade(p1, p2):
    """Require BOTH arms complete before honoring user's P2 tie preference."""
    if any(p['decision_150']['status']=='pending' for p in (p1,p2)):
        return dict(status='pending', chosen_arm=None)
    for arm, report in [('P2',p2),('P1',p1)]:
        if report['decision_150']['eligible_for_400']:
            return dict(status='selected',chosen_arm=arm,total_steps=400,
                        continuation_steps=250, restart_from_sft=False)
    return dict(status='neither_qualified',chosen_arm=None)


def markdown(report):
    pct=lambda x:'缺失' if x is None else f'{100*x:.2f}%'
    num=lambda x:'缺失' if x is None else f'{x:.4g}'
    lines=['# P1/P2 短验证报告','',f"运行：`{report['source_run']}`",'',
           '按 25 步窗口汇总；所有比例按原始计数加权，不平均每步百分比。仅使用固定验证集。','',
           '|窗口|臂|全错题不可解预算|全错题救回率|前3步预算|前3步平均ρ|全错/混合/全对题步|KL均值/最大|B梯度均值/最大|秒/步|验证success / pass@4|',
           '|---|---|---:|---:|---:|---:|---|---|---|---:|---|']
    for window in report['windows']:
        for key in ('pilot','S05'):
            s=window[key];af=s['per_group']['all_failure'];d=s['diagnostics']
            v=window['validation' if key=='pilot' else 'S05_validation']
            counts='/'.join(str(s['per_group'][g]['task_count']) for g in GROUPS)
            vals=[f"{window['start']}–{window['end']}",key,pct(af['dead_actual_budget_share']),pct(af['rescue_rate']),pct(af['early_expected_budget_share']),num(af['early_mean_rho']),counts,
                  num(d['kl']['mean'])+'/'+num(d['kl']['max']),num(d['part_b_grad_norm']['mean'])+'/'+num(d['part_b_grad_norm']['max']),num(d['step_seconds']['mean']),
                  '缺失' if v is None else pct(v['success_rate'])+' / '+pct(v['pass_at_4'])]
            lines.append('|'+ '|'.join(vals)+'|')
    decision=report['decision_150']
    lines.extend(['',f"150 步判定：**{decision['status']}**；逐项：`{json.dumps(decision['checks'])}`。",
                  f"验证第100/125/150步均值：{pct(decision['validation_last3_mean'])}；S05：{pct(decision['baseline_validation_last3_mean'])}；允许下限：{pct(decision['validation_threshold'])}。",
                  '', 'S05 是 v2，P1/P2 是 v2.2；该历史对照用于筛选，不代表仅门控的单变量因果比较。',
                  '门控不保证完整 PPO+KL 更新无偏。a 接近零是门控实现检查，不能单独当作效果改善。',
                  '无分叉或无全错题导致分母为零时记缺失，不按0通过。逐步组数、预期/实际预算、异常和来源见同名JSON。'])
    return '\n'.join(lines)+'\n'


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('reference');p.add_argument('--run',required=True);p.add_argument('--audit-candidates',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('report');p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--upto',type=int,default=150)
    p=sub.add_parser('choose');p.add_argument('--p1',required=True);p.add_argument('--p2',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='reference':
        result=build_reference(args.run,args.audit_candidates)
    elif args.command=='report':
        result=report_run(args.run,json.loads(Path(args.reference).read_text()),args.upto)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).with_suffix('.md').write_text(markdown(result))
    else:
        result=choose_upgrade(json.loads(Path(args.p1).read_text()),json.loads(Path(args.p2).read_text()))
    write_json(args.output,result)
    print(json.dumps({'output':args.output,'status':result.get('status',result.get('decision_150',{}).get('status','reference_ready'))}))


if __name__=='__main__':
    main()
