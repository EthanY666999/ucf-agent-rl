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
    configs=[load_config(ROOT/f'configs/4b/sokoban/{name}_seed42.json') for name in ['grpo4','tree_random_k2','ucf_v2']]
    assert len({c['model'] for c,_ in configs})==1
    for c,m in configs:
        assert (c['steps'],c['train_batch_size'],c['learning_rate'],c['eval_every'])==(400,64,1e-6,25)
        assert c['model'].endswith('sft_seed42/final_model')
    assert configs[1][0]['branch_interval']==2
    assert configs[2][0]['ucf_beta']==.25
    assert configs[2][0]['ucf_version']=='v2'


def test_unaccepted_ports_fail_closed():
    _,m=load_config(ROOT/'configs/9b/alfworld/ucf_v2.1_seed42.json')
    assert set(m['gates'])=={'distributed_training_backend_required','alfworld_training_integration_required','v2.1_2b_acceptance_required'}
    _,m=load_config(ROOT/'configs/4b/sokoban/nosft_ucf_v2.1_seed42.json')
    assert m['gates']==['v2.1_2b_acceptance_required']


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
    (tmp_path/'code/train.py').write_text('source = 1\n')
    before=verify_release.inventory()
    (tmp_path/'code/artifacts').mkdir()
    (tmp_path/'code/artifacts/.run.lock').write_text('runtime lock')
    assert verify_release.inventory()==before
    (tmp_path/'code/train.py').write_text('source = 2\n')
    assert verify_release.inventory()!=before
