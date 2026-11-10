import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from v3_manager import reconcile,read

def test_recovery_preserves_uncheckpointed_metrics_and_rollouts(tmp_path):
 p=tmp_path/'train_metrics.jsonl';original=''.join(json.dumps(dict(step=i))+'\n' for i in range(1,5));p.write_text(original)
 (tmp_path/'branch_events.jsonl').write_text(original)
 (tmp_path/'rollouts_0004.jsonl.gz').write_bytes(b'evidence')
 (tmp_path/'rollouts_0002.jsonl.gz').write_bytes(b'kept')
 r=reconcile(tmp_path,2)
 assert [json.loads(x)['step'] for x in p.read_text().splitlines()]==[1,2]
 archives=list((tmp_path/'recovery_archive').iterdir());assert len(archives)==1
 assert (archives[0]/'train_metrics.jsonl').read_text()==original
 assert (archives[0]/'rollouts_0004.jsonl.gz').read_bytes()==b'evidence'
 assert (tmp_path/'rollouts_0002.jsonl.gz').read_bytes()==b'kept'
 assert r['checkpoint_step']==2

def test_recovery_rejects_missing_durable_metric_before_mutating(tmp_path):
 p=tmp_path/'train_metrics.jsonl';p.write_text('{"step":1}\n{"step":3}\n')
 with pytest.raises(AssertionError):reconcile(tmp_path,3)
 assert p.read_text()=='{"step":1}\n{"step":3}\n'
 assert not (tmp_path/'recovery_archive').exists()

def test_recovery_archives_partial_json_line(tmp_path):
 p=tmp_path/'train_metrics.jsonl';p.write_text('{"step":1}\n{"step":')
 reconcile(tmp_path,1)
 assert p.read_text()=='{"step":1}\n'

def test_boundary_hold_never_launches_child(tmp_path,monkeypatch):
 import v3_manager as m
 state=tmp_path/'state';state.mkdir();(state/'hold_after_current.json').write_text('{}')
 locks=[tmp_path/'q.lock',tmp_path/'g.lock']
 plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(state=str(state),locks=[str(p) for p in locks],entries=[dict(id='next')])) )
 for k in ('UCF_INHERITED_QUEUE_LOCK_FD','UCF_INHERITED_GPU_LOCK_FD'):monkeypatch.delenv(k,raising=False)
 monkeypatch.setattr(sys,'argv',['v3_manager.py',str(plan)])
 monkeypatch.setattr(m,'run_entry',lambda *a:pytest.fail('Must not start a GPU child'))
 m.main()
 assert read(state/'boundary_held.json')['training_interrupted'] is False

def test_disk_guard_preserves_prior_checkpoint(tmp_path,monkeypatch):
 import ast,shutil
 from types import SimpleNamespace as NS
 source=Path(__file__).resolve().parents[1]/'code/train.py'
 module=ast.parse(source.read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
 namespace=dict(Path=Path,torch=NS(is_tensor=lambda x:False))
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
 cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'previous valid checkpoint')
 actor=NS(parameters=lambda:[NS(numel=lambda:100,element_size=lambda:4)])
 monkeypatch.setattr(shutil,'disk_usage',lambda p:NS(free=0))
 with pytest.raises(OSError,match='Checkpoint disk reserve'):
  namespace['save_checkpoint'](cp,actor,NS(state={}),1,None,{'ucf_v3':True},0)
 assert cp.read_bytes()==b'previous valid checkpoint' and not cp.with_suffix('.tmp').exists()

def test_scoped_admission_requires_backward_receipt_and_exact_helper(tmp_path):
 from v3_manager import numerical_admission,sha
 root=tmp_path/'acceptance';root.mkdir()
 helper=tmp_path/'scope.py';helper.write_text('verified helper')
 entry=tmp_path/'entry.py';entry.write_text('verified entrypoint')
 plan=dict(scoped_helper=str(helper),train_entrypoint=str(entry),source_sha256={str(helper):sha(helper),str(entry):sha(entry)})
 e=dict(kind='train',requires_scoped_gpu_acceptance=str(root))
 cfg=dict(vllm_batch_invariant_scoped=True,environment='sokoban')
 (root/'acceptance.json').write_text('{"passed":true}')
 (root/'config.json').write_text('{"environment":"sokoban"}')
 with pytest.raises(FileNotFoundError):numerical_admission(plan,e,cfg)
 scope=dict(passed=True,post_update_inference_then_backward=True,native_fp32_and_bf16_autocast_checked=True,helper_sha256=sha(helper))
 (root/'scope_acceptance.json').write_text(json.dumps(scope))
 numerical_admission(plan,e,cfg)
 helper.write_text('changed after GPU acceptance')
 with pytest.raises(AssertionError):numerical_admission(plan,e,cfg)
