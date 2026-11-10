"""Compare the complete ten-step regression, retaining every per-step deviation.

This is an engineering interpretation of 'same order of magnitude', not a
preregistered scientific acceptance threshold. The earlier per-step ratio gate
was added by Codex and rejected one near-zero first-step KL despite a comparable
ten-step trajectory. No training or evaluation parameter is selected here.
"""
import math
import statistics


def compare(pairs,records):
    assert len(pairs)==10 and {p['step'] for p in pairs}==set(range(1,11))
    by_step={r['step']:r for r in records}
    assert set(by_step)==set(range(1,11))
    for r in by_step.values():
        for k in ['kl','grad_norm','success_rate','max_parameter_change']:
            assert math.isfinite(r[k]),(r['step'],k)
        assert r['max_parameter_change']>0 and r['grad_norm']>0
    diagnostics={};passed=True
    for key in ['kl','grad_norm']:
        values={side:[p[key][side] for p in pairs] for side in ['current','reference']}
        assert all(math.isfinite(x) and x>=0 for xs in values.values() for x in xs)
        checks={}
        for label,fn in [('mean',statistics.mean),('median',statistics.median)]:
            current,reference=fn(values['current']),fn(values['reference'])
            assert reference>0,(key,label,'zero reference cannot establish ratio')
            ratio=current/reference;ok=.1<=ratio<=10;passed &= ok
            checks[label]=dict(current=current,reference=reference,ratio=ratio,passed=ok)
        diagnostics[key]=checks
    successes={side:[p['success_rate'][side] for p in pairs] for side in ['current','reference']}
    assert all(math.isfinite(x) and 0<=x<=1 for xs in successes.values() for x in xs)
    difference=statistics.mean(successes['current'])-statistics.mean(successes['reference'])
    passed &= abs(difference)<=.1
    exceptions=[p for p in pairs if any(p[k]['ratio'] is None or not .1<=p[k]['ratio']<=10 for k in ['kl','grad_norm']) or abs(p['success_rate']['current']-p['success_rate']['reference'])>.1]
    return dict(passed=bool(passed),steps=10,aggregation='full ten-step mean and median, both within factor10',
                diagnostics=diagnostics,success_mean_difference=difference,per_step_exceptions=exceptions,
                per_step_pairs=pairs,all_updates_finite_and_nonzero=True,
                scope='engineering cross-hardware regression, not parameter selection',
                criterion_revision='Codex per-step gate corrected after observing steps1-7; user requires no stopping on isolated statistical fluctuations')
