"""CPU regression for diagnostic cadence, with no production imports or GPU."""
import ast
import json
import os
from pathlib import Path
import time


def updater(tmp_path, monkeypatch):
    source=Path(__file__).resolve().parents[1]/'code/train.py'
    tree=ast.parse(source.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='ucf_update')
    calls=[]
    def component(*args):
        calls.append(args[-1]);return {'grad_norm':1.,'loss':1.}
    def update(*args):
        calls.append('real_update');return {'grad_norm':2.}
    ns=dict(json=json,Path=Path,os=os,time=time,component_grad=component,original_update=update)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    monkeypatch.setenv('PATR_PROGRESS_PATH',str(tmp_path/'progress.json'))
    return ns['ucf_update'],calls


def test_v3_unmeasured_is_missing_not_zero_and_still_updates(tmp_path,monkeypatch):
    update,calls=updater(tmp_path,monkeypatch)
    for step,expected in [(1,True),(2,False),(24,False),(25,True),(26,False),(50,True)]:
        calls.clear();(tmp_path/'status.json').write_text(json.dumps({'current_step':step}))
        result=update(None,None,None,[],{'ucf_v3':True,'eval_every':25},'ucf',None,None,64)
        assert calls==(['A','B','real_update'] if expected else ['real_update'])
        assert result['grad_norm']==2.
        assert result['ucf_component_diagnostics_computed']==expected
        assert (result['ucf_preupdate_fullbatch_components'] is not None)==expected


def test_non_v3_keeps_measurements_without_requiring_status(tmp_path,monkeypatch):
    update,calls=updater(tmp_path,monkeypatch)
    result=update(None,None,None,[],{},'ucf',None,None,64)
    assert calls==['A','B','real_update']
    assert result['ucf_component_diagnostics_computed'] is True
