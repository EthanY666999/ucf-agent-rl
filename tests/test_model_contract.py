import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_model_tokenizer_templates_and_shapes():
    spec = importlib.util.spec_from_file_location("model_contract", ROOT / "scripts/check_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(ROOT / "data/model_metadata")
    assert report['transformers']=='5.22.2'
    assert report["family_contract_identical"]
    assert {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"} <= report["models"].keys()
    assert not report["weights_downloaded"]
    for model in report['models'].values():
        actual = model['actual_ucf_action_ids']
        assert set(actual) == {'Left', ' Left', 'Down', ' Down', 'Right', ' Right', 'Up', ' Up'}
        assert len(set(actual.values())) == 8
        assert 'left' in model['parser_diagnostic_action_ids']
        assert 'left' not in actual


@pytest.mark.parametrize('size', ['4B', '9B'])
def test_bootstrap_uses_explicit_same_size_metadata(size, tmp_path):
    spec = importlib.util.spec_from_file_location('tested_q35_loader', ROOT / 'code/scripts/q35_engine_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    base = ROOT / 'data/model_metadata' / f'Qwen3.5-{size}'
    saved = json.loads((base / 'config.json').read_text())['text_config']
    exported = tmp_path / 'text_export'
    exported.mkdir()
    (exported / 'config.json').write_text(json.dumps(saved))
    assert loader.engine_source(exported, base=base) == base
    with pytest.raises(ValueError, match='explicit bootstrap_model'):
        loader.engine_source(exported)
    with pytest.raises(AssertionError, match='architecture differs'):
        loader.engine_source(exported, base=ROOT / 'data/model_metadata/Qwen3.5-2B')
    calls = []
    actor, engine = object(), object()
    def factory(**kwargs):
        calls.append(('factory', kwargs))
        return engine
    def sync(a, e):
        assert a is actor and e is engine
        calls.append(('sync', None))
    assert loader.build_engine(actor, exported, factory, sync, base=base) is engine
    assert [call[0] for call in calls] == ['factory', 'sync']
    assert calls[0][1] == {'model': str(base), 'tokenizer': str(exported)}
