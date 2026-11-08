"""Admission tests use tiny temporary files; never load real model weights."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiment import load_config
from model_identity import identity,verify_sft


def test_independent_same_sft_start_and_fixed_protocol():
    configs=[load_config(ROOT/f'configs/4b/sokoban/{name}_seed42.json') for name in ['grpo4','tree_random_k2','ucf_v2.2']]
    assert len({c['model'] for c,_ in configs})==1
    for c,m in configs:
        assert (c['steps'],c['train_batch_size'],c['learning_rate'],c['eval_every'])==(400,64,1e-6,25)
        assert c['model'].endswith('sft_seed42/final_model')
    assert configs[1][0]['branch_interval']==2
    assert configs[2][0]['ucf_beta']==.25
    assert configs[2][0]['ucf_version']=='v2.2'
    assert configs[2][0]['ucf_controller']=='leave_one_task_out'


def test_unaccepted_ports_fail_closed():
    _,m=load_config(ROOT/'configs/9b/alfworld/ucf_v2.1_seed42.json')
    assert set(m['gates'])=={'distributed_training_backend_required','alfworld_training_integration_required','v2.1_2b_acceptance_required'}
    c,m=load_config(ROOT/'configs/4b/sokoban/nosft_ucf_v2.2-ff_seed42.json')
    assert m['gates']==['v2.2-ff_2b_acceptance_and_source_import_required']
    assert c['turn_stop_protocol']=='first_complete_action_block_token_boundary'
    assert c['ucf_fork_point']=='immediately_after_first_opening_fence'


@pytest.mark.parametrize('version',['v2','v2.1','v2.2','v2.2-ff'])
def test_no_ucf_launch_without_verified_import_even_if_config_gates_removed(version,tmp_path,monkeypatch):
    import run_experiment
    c=json.loads((ROOT/'configs/4b/sokoban/ucf_v2.2_seed42.json').read_text())
    c['ucf_version']=version;c['_release']['gates']=[]
    path=tmp_path/'ucf.json';path.write_text(json.dumps(c))
    monkeypatch.setattr(sys,'argv',['run_experiment.py','--config',str(path),'--execute'])
    def prohibited(*args,**kwargs):
        raise AssertionError('A blocked method reached subprocess execution')
    monkeypatch.setattr(run_experiment.subprocess,'run',prohibited)
    with pytest.raises(SystemExit,match='historical source only|awaits verified 2B'):
        run_experiment.main()


def test_direct_training_cannot_bypass_ucf_release_gate(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('direct_train_gate_test',ROOT/'code/train.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    path=tmp_path/'legacy.json';path.write_text(json.dumps(dict(ucf_version='v2')))
    monkeypatch.setattr(sys,'argv',['train.py','--method','ucf','--config',str(path)])
    with pytest.raises(RuntimeError,match='historical source only'):
        train.main()


def test_flh_has_no_default_sft_dataset_and_all_dependents_are_blocked():
    for path in (ROOT/'configs/4b/flh').glob('*.json'):
        cfg,meta=load_config(path)
        assert meta['sft_data'] is None
        if meta['stage']=='sft' or meta['start']=='sft':
            assert 'flh_sft_start_pending_2b_H01_decision' in meta['gates']


def test_active_plan_remains_4b_and_method_order_is_preserved():
    plan=json.loads((ROOT/'configs/plan.json').read_text())
    assert all(x['id'].startswith('4b_') for x in plan['main']+plan['conditional'])
    assert all(x['id'].startswith('9b_') for x in plan['deferred_unchanged'])
    phases=[x['id'] for x in plan['main']]
    assert phases.index('4b_sokoban_ucf_v2.2_seed42') < phases.index('4b_flh_base_eval_seed42')
    assert phases.index('4b_flh_ucf_v2.2_seed42') < phases.index('4b_sokoban_starpo_s_seed42')
    assert phases.index('4b_flh_gigpo_seed42') < phases.index('4b_sokoban_grpo4_seed43')
    for path in (ROOT/'configs/4b').rglob('*.json'):
        cfg=json.loads(path.read_text())
        assert (cfg['micro_batch_size'],cfg['micro_batch_token_limit'],cfg['vllm_memory_fraction'],
                cfg['vllm_max_sequences'],cfg['optimizer'])==(1,32768,.12,8,'adamw')
        assert cfg.get('ucf_version') not in {'v2','v2.1'}


def test_changed_sft_weights_rejected(tmp_path):
    import hashlib
    m=tmp_path/'final_model';m.mkdir()
    (m/'model.safetensors').write_bytes(b'tiny identity fixture, never deserialized')
    (m/'config.json').write_text('{}')
    (tmp_path/'status.json').write_text(json.dumps(dict(status='complete',steps=63)))
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'evaluation/status.json').write_text(json.dumps(dict(status='evaluation_complete')))
    evidence={'status.json':hashlib.sha256((tmp_path/'status.json').read_bytes()).hexdigest()}
    (tmp_path/'sft_acceptance.json').write_text(json.dumps(dict(passed=True,model_identity=identity(m),evidence_sha256=evidence)))
    verify_sft(tmp_path)
    (m/'model.safetensors').write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='model changed'):verify_sft(tmp_path)


def test_export_excludes_weights_and_raw_rollouts():
    from export_results import allowed
    assert allowed(Path('ucf_logs/select_0001.jsonl'))
    assert allowed(Path('additional_metrics.json'))
    for name in ['checkpoint.pt','final_model/model.safetensors','rollouts_0001.jsonl.gz','ucf_logs/raw_leaves_0001.jsonl.gz']:
        assert not allowed(Path(name))


def test_runtime_lock_does_not_invalidate_source_manifest(tmp_path,monkeypatch):
    import verify_release
    monkeypatch.setattr(verify_release,'ROOT',tmp_path)
    for tree in verify_release.TREES:(tmp_path/tree).mkdir()
    for name in verify_release.DEPENDENCIES:(tmp_path/name).write_text('transformers==5.22.2\n')
    (tmp_path/'code/train.py').write_text('source = 1\n')
    before=verify_release.inventory()
    (tmp_path/'code/artifacts').mkdir()
    (tmp_path/'code/artifacts/.run.lock').write_text('runtime lock')
    assert verify_release.inventory()==before
    (tmp_path/'requirements-cpu.txt').write_text('transformers==5.5.4\n')
    assert verify_release.inventory()!=before
    (tmp_path/'requirements-cpu.txt').write_text('transformers==5.22.2\n')
    (tmp_path/'code/train.py').write_text('source = 2\n')
    assert verify_release.inventory()!=before
