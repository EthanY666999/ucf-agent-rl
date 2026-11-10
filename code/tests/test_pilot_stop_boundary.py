"""Execute the real startup boundary without importing torch/vLLM or using GPUs."""
import ast
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

CODE=Path(__file__).resolve().parents[1]


def startup_fragment():
    main=next(n for n in ast.parse((CODE/'train.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='main')
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    begin=next(i for i,n in enumerate(main.body) if assignment(n,'validation'))
    end=next(i for i,n in enumerate(main.body) if i>begin and assignment(n,'start'))
    fragment=main.body[begin:end]
    # Mark the point where the production code would begin its next update.
    fragment+=ast.parse("events.append('would_enter_training'); return 'continue'").body
    fn=ast.FunctionDef(name='startup',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                       body=fragment,decorator_list=[])
    return compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_train_startup','exec')


@pytest.mark.parametrize('checkpoint_step,limit,expect_stop',[
    (150,150,True),(151,150,True),(149,150,False),(150,None,False),
])
def test_resumed_stop_boundary_recovers_validation_without_next_update_or_final_test(
        tmp_path,monkeypatch,checkpoint_step,limit,expect_stop):
    events=[];cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'preserved-checkpoint')
    def evaluate(*args):
        assert args[-1]=='validation', 'Short pilot must not evaluate the test set'
        events.append('validation');Path(args[4]).write_text('{}')
        return {'success_rate':.5}
    def recover(out,step,cfg,validation,run):
        events.append('recovery');run(out/f'validation_{step:04d}.json')
    module=types.ModuleType('scripts.validation_recovery');module.complete_checkpoint_validation=recover
    monkeypatch.setitem(sys.modules,'scripts.validation_recovery',module)
    def atomic(path,value):
        events.append('status');path.write_text(json.dumps(value))
    scope=dict(args=NS(resume=True,stop_after_step=limit,smoke=False,method='ucf'),
               cfg={'steps':400},data={'validation':[{'uid':'fixed-validation'}]},
               start_step=checkpoint_step,out=tmp_path,engine=None,tokenizer=None,
               evaluate=evaluate,atomic_json=atomic,events=events)
    exec(startup_fragment(),scope);result=scope['startup']()
    assert events[:2]==['recovery','validation']
    assert cp.read_bytes()==b'preserved-checkpoint'
    if expect_stop:
        assert result is None and events==['recovery','validation','status']
        saved=json.loads((tmp_path/'status.json').read_text())
        assert saved['status']=='paused_checkpointed' and saved['step']==checkpoint_step
        assert saved['stop_after_step']==150 and not saved['final_evaluation_pending']
    else:
        assert result=='continue' and events==['recovery','validation','would_enter_training']
        assert not (tmp_path/'status.json').exists()


def test_default_fresh_run_startup_is_unchanged(tmp_path):
    events=[]
    def evaluate(*args):events.append('initial_validation');return {}
    scope=dict(args=NS(resume=False,stop_after_step=None,smoke=False,method='ucf'),
               cfg={'steps':400},data={'validation':[{}]},start_step=0,out=tmp_path,
               engine=None,tokenizer=None,evaluate=evaluate,events=events,json=json)
    exec(startup_fragment(),scope)
    assert scope['startup']()=='continue'
    assert events==['initial_validation','would_enter_training']
