"""CPU-only safety tests: no SSH, GPU import, or production queue writes."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_p1p2_supervisor as controller
import pilot_return_queue as returning


def write(path,value):
    controller.atomic(path,value)
    return path


def paused_run(root):
    root.mkdir(parents=True,exist_ok=True)
    write(root/'status.json',dict(status='paused_checkpointed',step=150,total_steps=400))
    (root/'checkpoint.pt').write_bytes(b'CPU fixture, not a model')
    (root/'train_metrics.jsonl').write_text(''.join(json.dumps(dict(step=i,loss=0.,kl=.1,grad_norm=.2,step_seconds=1.))+'\n' for i in range(1,151)))
    for i in range(0,151,25):
        write(root/f'validation_{i:04d}.json',dict(split='validation',n_tasks=100,n_trajectories=400))
    write(root/'pilot150_verification.json',controller.pilot_verified(root))
    return root


def entry(root,ident='P1'):
    return dict(id=ident,output=str(root),command=['python','/isolated/code/train.py','--method','ucf'])


def test_pid_identity_uses_start_ticks_and_handles_prelaunch_journal(tmp_path):
    proc=tmp_path/'proc'; path=proc/'123';path.mkdir(parents=True)
    fields=['S']+['0']*18+['789']+['0']*3
    (path/'stat').write_text('123 (trainer worker) '+' '.join(fields))
    assert controller.same_process(dict(pid=123,start_ticks='789'),proc)
    assert not controller.same_process(dict(pid=123,start_ticks='790'),proc)
    assert not controller.same_process(dict(started=1),proc)
    assert not controller.same_process(dict(pid=999,start_ticks='789'),proc)


def test_release_fence_checks_exact_hash_and_preserves_unknown_marker(tmp_path):
    state=tmp_path/'state';state.mkdir()
    fence=write(tmp_path/'needs_manual.json',dict(handoff_token='authorized'))
    plan=dict(boundary_fence=dict(path=str(fence),sha256=controller.sha(fence),token='authorized'))
    expected=copy.deepcopy(plan)
    write(fence,dict(handoff_token='someone_else',reason='do not remove'))
    with pytest.raises(AssertionError):controller.archive_control_fence(plan,state)
    assert fence.exists()
    write(fence,dict(handoff_token='authorized'))
    controller.archive_control_fence(expected,state)
    controller.archive_control_fence(expected,state)
    assert not fence.exists()


def test_pilot_receipt_rejects_final_test_duplicates_and_missing_checkpoint(tmp_path):
    out=paused_run(tmp_path/'P1')
    assert controller.pilot_verified(out)['final_test_used'] is False
    write(out/'final_test.json',dict(success_rate=1.))
    with pytest.raises(AssertionError):controller.pilot_verified(out)
    (out/'final_test.json').unlink()
    original=(out/'train_metrics.jsonl').read_text()
    (out/'train_metrics.jsonl').write_text(original+original.splitlines()[-1]+'\n')
    with pytest.raises(AssertionError):controller.pilot_verified(out)
    (out/'train_metrics.jsonl').write_text(original)
    (out/'checkpoint.pt').unlink()
    with pytest.raises(AssertionError):controller.pilot_verified(out)


def test_stop_and_resume_commands_preserve_base_command(tmp_path):
    e=entry(tmp_path)
    original=e['command'][:]
    assert controller.command_for(e,True)==original+['--stop-after-step','150']
    assert controller.command_for(e,False,True)==original+['--resume']
    assert controller.command_for(e,True,True)==original+['--stop-after-step','150','--resume']
    assert e['command']==original
    e['command']=['python','run_ucf.py']
    with pytest.raises(AssertionError):controller.command_for(e,True)


def fixture_queue(tmp_path):
    ids=['S05','F03','H01','H02','S06','H03','H04','H05','S13','S15','S07','S10','S16']
    return dict(entries=[dict(id=i,output=str(tmp_path/i),config=dict(seed=43 if i=='S10' else 42,
                    ucf_controller='v2',ucf_corrections='v22_question_loo_rejection256',fork_point='first_fence'),
                    config_difference=dict(ucf_corrections='x',fork_point='y')) for i in ids])


def test_return_wrapper_only_reorders_authorized_pair_and_restores_s10_v2(tmp_path):
    original=fixture_queue(tmp_path); frozen=copy.deepcopy(original)
    revised=returning.revised_plan(original)
    ids=[e['id'] for e in revised['entries']]
    assert ids[ids.index('S06'):ids.index('S06')+4]==['S06','S07','S10','H03']
    assert [x for x in ids if x not in ('S07','S10')]==[e['id'] for e in original['entries'] if e['id'] not in ('S07','S10')]
    for e in revised['entries']:
        if e['id']!='S10':
            assert e['config']==next(x['config'] for x in original['entries'] if x['id']==e['id'])
    s10=next(e for e in revised['entries'] if e['id']=='S10')
    assert s10['config']==dict(seed=43,ucf_controller='v2')
    assert original==frozen


def test_failed_original_entries_are_fenced_not_retried_by_return_launcher(tmp_path):
    original=fixture_queue(tmp_path)
    for ident in ('S07','S10'):
        write(tmp_path/ident/'needs_manual.json',dict(reason='terminal preserved failure'))
    revised=returning.revised_plan(original,dict(S07=False,S10=False))
    by_id={e['id']:e for e in revised['entries']}
    assert 'strict_predecessor' not in by_id['S07']
    assert 'strict_predecessor' not in by_id['S10']
    assert by_id['H03']['strict_predecessor']==str(tmp_path/'S06')
    (tmp_path/'S10/needs_manual.json').unlink()
    with pytest.raises(AssertionError):returning.revised_plan(original,dict(S07=False,S10=False))


def test_resume_only_accepts_exact_checkpoint_metric_prefix(tmp_path,monkeypatch):
    out=paused_run(tmp_path/'P1');e=entry(out)
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**k:'150\n')
    assert controller.recovery_step(dict(python='python'),e,True)==150
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**k:'125\n')
    with pytest.raises(AssertionError,match='checkpoint prefix'):
        controller.recovery_step(dict(python='python'),e,True)
    assert len((out/'train_metrics.jsonl').read_text().splitlines())==150


def test_promoted_restart_recognizes_pilot_receipt_without_old_checkpoint(tmp_path):
    out=paused_run(tmp_path/'P2');state=tmp_path/'state';e=entry(out,'P2')
    write(state/'S05b_promotion.json',dict(selected_arm='P2',output=str(out)))
    write(state/'reports/P2/step_0150.json',dict(decision_150=dict(status='pass',eligible_for_400=True)))
    write(out/'status.json',dict(status='complete',step=400,total_steps=400))
    (out/'checkpoint.pt').unlink()
    assert controller.prior_pilot_receipt(state,e)
    assert controller.run_entry({},state,e,None,pilot=True)


def test_decision_prefers_p2_and_missing_report_fails_closed(tmp_path):
    for ident in ('P1','P2'):
        write(tmp_path/'reports'/ident/'step_0150.json',dict(decision_150=dict(status='pass',eligible_for_400=True)))
    assert controller.decide({},tmp_path,dict(P1=True,P2=True))['chosen_arm']=='P2'
    (tmp_path/'decision.json').unlink()
    assert controller.decide({},tmp_path,dict(P1=False,P2=True))['chosen_arm']=='P2'
    (tmp_path/'decision.json').unlink();(tmp_path/'reports/P2/step_0150.json').unlink()
    assert controller.decide({},tmp_path,dict(P1=True,P2=True))['status']=='pending'


def test_verified_migration_preserves_logical_path_and_all_bytes(tmp_path):
    out=paused_run(tmp_path/'home/P2');state=tmp_path/'state';state.mkdir()
    plan=dict(promotion_storage_root=str(tmp_path/'data'),minimum_free_bytes=0)
    before=controller.tree_hashes(out)
    assert controller.relocate_paused_run(plan,state,entry(out,'P2'))
    assert out.is_symlink() and out.resolve()==tmp_path/'data/P2'
    assert controller.tree_hashes(out)==before
    assert not out.with_name(out.name+'.verified_migration_source').exists()
    assert controller.relocate_paused_run(plan,state,entry(out,'P2'))


def test_migration_no_space_keeps_source_and_checkpoint(tmp_path,monkeypatch):
    out=paused_run(tmp_path/'home/P1');state=tmp_path/'state';state.mkdir()
    monkeypatch.setattr(controller.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    assert not controller.relocate_paused_run(dict(promotion_storage_root=str(tmp_path/'data'),minimum_free_bytes=1),state,entry(out))
    assert out.is_dir() and not out.is_symlink() and (out/'checkpoint.pt').exists()
    assert not (tmp_path/'data/P1').exists()


def test_migration_partial_copy_never_deletes_source(tmp_path):
    out=paused_run(tmp_path/'home/P1');state=tmp_path/'state';state.mkdir()
    target=tmp_path/'data/P1';target.mkdir(parents=True)
    (target/'checkpoint.pt').write_bytes(b'partial transfer')
    assert not controller.relocate_paused_run(dict(promotion_storage_root=str(tmp_path/'data'),minimum_free_bytes=0),state,entry(out))
    assert not out.is_symlink() and (out/'checkpoint.pt').read_bytes()!=b'partial transfer'
    assert target.exists()


def test_live_child_prevents_migration_and_duplicate_training(tmp_path,monkeypatch):
    out=paused_run(tmp_path/'home/P1');state=tmp_path/'state'
    write(state/'attempts/P1.json',[dict(pid=123,start_ticks='x')])
    monkeypatch.setattr(controller,'same_process',lambda record:True)
    with pytest.raises(AssertionError,match='live training'):
        controller.relocate_paused_run(dict(promotion_storage_root=str(tmp_path/'data'),minimum_free_bytes=0),state,entry(out))
    assert not out.is_symlink()


def test_failed_arm_preserves_marker_but_admission_drift_stops_everything(tmp_path,monkeypatch):
    out=tmp_path/'P1';state=tmp_path/'state';e=entry(out)
    def fail(*a,**k):raise RuntimeError('test failed')
    monkeypatch.setattr(controller,'run_entry',fail)
    assert not controller.guarded_run({},state,e,None,True)
    assert (out/'needs_manual.json').exists()
    def corrupt(*a,**k):raise controller.AdmissionError('source changed')
    monkeypatch.setattr(controller,'run_entry',corrupt)
    with pytest.raises(controller.AdmissionError):controller.guarded_run({},state,e,None,True)


def test_migration_recovers_after_verified_source_removed_before_final_journal(tmp_path):
    out=paused_run(tmp_path/'home/P2');state=tmp_path/'state';state.mkdir()
    plan=dict(promotion_storage_root=str(tmp_path/'data'),minimum_free_bytes=0)
    assert controller.relocate_paused_run(plan,state,entry(out,'P2'))
    path=state/'storage/P2.json';record=controller.read(path)
    record['status']='linked_verified';write(path,record)
    assert controller.relocate_paused_run(plan,state,entry(out,'P2'))
    assert controller.read(path)['status']=='complete'


def readiness_fixture(tmp_path):
    state=tmp_path/'state';source=tmp_path/'runtime/code';runs=tmp_path/'runs'
    source.mkdir(parents=True)
    (source/'train.py').write_text('# CPU test fixture\n')
    write(source/'SOURCE_MANIFEST_q35.json',dict(source='test'))
    write(source/'artifacts/frozen_q35.json',dict(source_sha256={'train.py':controller.sha(source/'train.py')}))
    for name in ('cpu_acceptance','gpu_acceptance','calibration','phase1_gate'):
        write(source/'artifacts/q35_gates'/f'{name}.json',dict(passed=True,model_family='Qwen3.5'))
    admissions=[]
    paths=[runs/'hard_queue/S02_SFT_seed42/verification.json',runs/'datasets/sokoban/cpu_acceptance.json',
           source/'artifacts/hard_cpu_acceptance.json',runs/'stage0/sokoban_acceptance/acceptance.json']
    paths += [runs/'queue_q35_hard/ucf_v22_20261109'/name for name in ('cpu_functional.json','htsim_first_fence.json','prefix_causality.json')]
    for i,path in enumerate(paths):
        write(path,dict(passed=True));admissions.append(dict(role='gate'+str(i),path=str(path),sha256=controller.sha(path)))
    entries=[]
    for ident in ('P1','P2','S07','S10'):
        cfg=dict(steps=400,eval_every=25,model=str(runs/'hard_queue/S02_SFT_seed42/sft/final_model'),seed=43 if ident=='S10' else 42)
        if ident in ('P1','P2'):cfg.update(solvability_gate=True,early_exploration_floor=ident=='P2',ucf_corrections='v22_question_loo_rejection256')
        if ident=='S10':cfg['ucf_controller']='v2'
        path=write(state/'configs'/f'{ident}.json',cfg)
        entries.append(dict(id=ident,config_path=str(path),output=str(tmp_path/'output'/ident)))
    plan=dict(schema_version=1,source_root=str(source),original_root=str(source),runs_root=str(runs),entries=entries)
    write(state/'pilot_plan.json',plan)
    cpu=write(state/'cpu.json',dict(passed=True))
    ready=dict(passed=True,pilot_plan_sha256=controller.sha(state/'pilot_plan.json'),
               files_sha256={e['config_path']:controller.sha(e['config_path']) for e in entries},
               cpu_acceptance=dict(path=str(cpu),sha256=controller.sha(cpu)),admission_receipts=admissions)
    for anchor in (source/'SOURCE_MANIFEST_q35.json',source/'artifacts/frozen_q35.json'):
        ready['files_sha256'][str(anchor)]=controller.sha(anchor)
    write(state/'ready.json',ready)
    return state,plan,ready


def test_admission_checks_bound_configs_family_source_and_inherited_gates(tmp_path):
    state,plan,ready=readiness_fixture(tmp_path)
    assert controller.verify_ready(state,check_git=False)==plan
    path=Path(plan['entries'][0]['config_path']);cfg=controller.read(path);cfg['steps']=150;write(path,cfg)
    with pytest.raises(AssertionError):controller.verify_ready(state,check_git=False)
    cfg['steps']=400;write(path,cfg)
    (Path(plan['source_root'])/'train.py').write_text('changed')
    with pytest.raises(AssertionError,match='Frozen pilot source'):
        controller.verify_ready(state,check_git=False)


def test_admission_cannot_drop_a_required_inherited_gate(tmp_path):
    state,plan,ready=readiness_fixture(tmp_path)
    ready['admission_receipts'].pop();write(state/'ready.json',ready)
    with pytest.raises(AssertionError,match='Missing inherited admission'):
        controller.verify_ready(state,check_git=False)


def test_final_pair_report_is_cpu_only_and_does_not_block_queue_on_failure(tmp_path,monkeypatch):
    plan=dict(python='python',original_root=str(tmp_path/'code'),runs_root=str(tmp_path/'runs'))
    def failure(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        assert command[-1].endswith('S05b_vs_S05_paired.json')
        raise RuntimeError('test pairing failure')
    monkeypatch.setattr(controller.subprocess,'run',failure)
    controller.paired_final_report(plan,tmp_path/'state',entry(tmp_path/'P2','P2'))
    assert controller.read(tmp_path/'state/paired_report_error.json')['selection_used_test'] is False
