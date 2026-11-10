import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from fork_budget_checkpoint import continuation_config,payload_sha
from v3_manager import reconcile,initial_step

def metadata(path):
    (path/'initialization.json').write_text(json.dumps(dict(kind='shared_rl_warmstart_budget_exploration',
        task_sft=False,not_a_fresh_raw_start_comparison=True,initial_step=16)))

def test_continuation_keeps_all_learning_parameters(tmp_path):
    cfg=dict(max_response_tokens=4096,ucf_v3=True,learning_rate=1e-6,ucf_beta=.25,
        max_turns=10,seed=42,ucf_log_dir='/previous/logs')
    for method in ('ucf','grpo'):
        result=continuation_config(cfg,method,tmp_path/method)
        assert result['max_response_tokens']==10240
        for k in ('learning_rate','ucf_beta','max_turns','seed'):assert result[k]==cfg[k]
    assert cfg['max_response_tokens']==4096

def test_payload_digest_excludes_config_but_catches_weight_and_rng_changes():
    import torch
    state=dict(model={'w':torch.tensor([1.,2.])},optimizer={'m':torch.tensor([.1,.2])},
        torch_rng=torch.tensor([1,2],dtype=torch.uint8),rng={'a':123},step=16,config={'budget':4096})
    before=payload_sha(state)
    changed=copy.deepcopy(state);changed['config']['budget']=10240
    assert payload_sha(changed)==before
    changed['model']['w'][0]+=1
    assert payload_sha(changed)!=before
    changed=copy.deepcopy(state);changed['rng']['a']+=1
    assert payload_sha(changed)!=before

def test_initial_fork_has_no_fabricated_metrics(tmp_path):
    metadata(tmp_path)
    assert initial_step(tmp_path)==16
    assert reconcile(tmp_path,16)['checkpoint_step']==16
    assert not (tmp_path/'train_metrics.jsonl').exists()

def test_fork_recovery_preserves_only_own_durable_updates(tmp_path):
    metadata(tmp_path)
    metrics=tmp_path/'train_metrics.jsonl'
    original=''.join(json.dumps({'step':i})+'\n' for i in (17,18,19))
    metrics.write_text(original)
    reconcile(tmp_path,18)
    assert [json.loads(x)['step'] for x in metrics.read_text().splitlines()]==[17,18]
    archived=list((tmp_path/'recovery_archive').glob('*/train_metrics.jsonl'))
    assert len(archived)==1 and archived[0].read_text()==original

def test_fork_recovery_cannot_invent_missing_warmstart_update(tmp_path):
    metadata(tmp_path)
    (tmp_path/'train_metrics.jsonl').write_text('{"step":18}\n')
    with pytest.raises(AssertionError):reconcile(tmp_path,18)
    assert not (tmp_path/'recovery_archive').exists()
