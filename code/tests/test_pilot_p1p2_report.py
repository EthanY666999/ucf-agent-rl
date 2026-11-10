import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('pilot_report',Path(__file__).parents[1]/'scripts/pilot_p1p2_report.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def counts(tasks=10,rescues=2,dead=0):
    return dict(task_count=tasks,rescued_tasks=rescues,candidate_count=100,
                sum_rho=10.,selected_count=10,dead_sum_rho=float(dead),
                dead_selected_count=dead,early_count=40,early_sum_rho=5.,
                early_selected_count=4)


def row(step=1,**overrides):
    selection={g:counts() for g in m.GROUPS}
    result=dict(step=step,selection=selection,kl=1.,part_b_grad_norm=1.,step_seconds=10.,invalid_metrics={})
    result.update(overrides)
    return result


def fixture():
    rows=[row(i) for i in range(1,151)]
    baseline=copy.deepcopy(rows)
    for r in baseline:r['selection']['all_failure']['rescued_tasks']=1
    val={str(s):dict(success_rate=.80,pass_at_4=.85) for s in m.EVAL_STEPS}
    ref=dict(rows=baseline,cumulative=m.aggregate(baseline),validation=copy.deepcopy(val),
             spike_thresholds={'kl':3.,'part_b_grad_norm':3.})
    return rows,val,ref


def test_ratios_use_pooled_task_counts_not_mean_of_step_ratios():
    a,b=row(1),row(2)
    a['selection']['all_failure']=counts(tasks=1,rescues=1)
    b['selection']['all_failure']=counts(tasks=9,rescues=0)
    result=m.aggregate([a,b],1,2)['per_group']['all_failure']
    assert result['rescue_rate']==.1
    assert result['early_mean_rho']==.125


def test_linear_quantile_matches_hand_calculated_noninteger_position():
    assert m.quantile([0,10,20],.99)==19.8


def test_complete_passing_arm_and_validation_three_point_mean():
    rows,val,ref=fixture()
    val=copy.deepcopy(val);val['100']['success_rate']=.75;val['125']['success_rate']=.78;val['150']['success_rate']=.81
    result=m.evaluate_150(rows,val,ref)
    assert result['status']=='pass'
    assert result['validation_last3_mean']==pytest.approx(.78)
    assert result['validation_threshold']==pytest.approx(.77)


def test_a_is_strict_less_than_ten_percent_and_b_strict_improvement():
    rows,val,ref=fixture()
    for r in rows:
        r['selection']['all_failure']['dead_selected_count']=1
        r['selection']['all_failure']['rescued_tasks']=1
    d=m.evaluate_150(rows,val,ref)
    assert d['status']=='fail'
    assert not d['checks']['a'] and not d['checks']['b']


def test_missing_step_or_validation_cannot_promote():
    rows,val,ref=fixture()
    assert m.evaluate_150(rows[:-1],val,ref)['status']=='pending'
    del val['125']
    d=m.evaluate_150(rows,val,ref)
    assert d['status']=='pending' and not d['eligible_for_400']


def test_empty_selection_denominator_not_misrepresented_as_zero_waste():
    rows,val,ref=fixture()
    for r in rows:r['selection']['all_failure']['selected_count']=0
    d=m.evaluate_150(rows,val,ref)
    assert d['all_failure']['dead_actual_budget_share'] is None
    assert not d['checks']['a'] and not d['eligible_for_400']


def test_nonfinite_diagnostic_is_preserved_as_invalid_not_silently_passed():
    rows,val,ref=fixture()
    raw={'step':8,'kl':float('nan'),'step_seconds':1.,'ucf_preupdate_fullbatch_components':{'B':{'grad_norm':1.}}}
    rows[7]=m.compact_step(raw,rows[7]['selection'])
    d=m.evaluate_150(rows,val,ref)
    assert not d['checks']['f']
    assert d['invalid_metrics'][0]['fields']['kl']=='nan'
    json.dumps(d,allow_nan=False)


def test_part_b_spike_is_distinguished_from_total_grad():
    rows,val,ref=fixture();rows[30]['part_b_grad_norm']=3.01
    d=m.evaluate_150(rows,val,ref)
    assert not d['checks']['f']
    assert d['spikes']==[dict(step=31,metric='part_b_grad_norm',value=3.01,threshold=3.)]


def test_choose_waits_for_p2_and_prefers_p2_if_both_qualify():
    passed={'decision_150':{'status':'pass','eligible_for_400':True}}
    pending={'decision_150':{'status':'pending','eligible_for_400':False}}
    failed={'decision_150':{'status':'fail','eligible_for_400':False}}
    assert m.choose_upgrade(passed,pending)['chosen_arm'] is None
    assert m.choose_upgrade(passed,passed)['chosen_arm']=='P2'
    assert m.choose_upgrade(passed,failed)['chosen_arm']=='P1'
    assert m.choose_upgrade(failed,failed)['status']=='neither_qualified'


def test_read_validation_rejects_test_split_and_never_reads_final_test(tmp_path):
    (tmp_path/'final_test.json').write_text('NOT JSON AND MUST NOT BE READ')
    assert m.read_validation(tmp_path,25) is None
    p=tmp_path/'validation_0025.json'
    p.write_text(json.dumps(dict(split='test',success_rate_pct=99,pass_at_4_pct=100)))
    with pytest.raises(ValueError,match='validation split'):m.read_validation(tmp_path,25)


def test_live_counts_keep_zero_probability_early_nodes_in_denominator():
    d={'pilot_solvability':{'per_group':{g:counts() for g in m.GROUPS}}}
    actual=m.selection_from_live(d)
    assert actual['all_failure']['early_count']==40
    assert actual['all_failure']['early_sum_rho']==5
    del d['pilot_solvability']['per_group']['mixed']['rescued_tasks']
    assert m.selection_from_live(d) is None


def test_partial_metrics_line_is_ignored_and_duplicate_step_rejected(tmp_path):
    p=tmp_path/'train_metrics.jsonl'
    p.write_text('{"step":1}\n{"step":2')
    assert list(m.read_metrics(tmp_path))==[1]
    p.write_text('{"step":1}\n{"step":1}\n')
    with pytest.raises(ValueError,match='duplicate'):m.read_metrics(tmp_path)
