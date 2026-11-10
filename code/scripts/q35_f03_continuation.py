"""User-authorized F03 compatibility evaluation and shared-path RL admission."""
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from q35_p0_plan import ROOT,RUNS,PYTHON,SFT
from q35_run_receipt import read,atomic,sha,rows,summarize
from q35_p0_supervisor import status,relocate_verified_model
STATE=RUNS/'queue_q35_hard'


def verify_authorized_base_gate(path):
    receipt=read(path)
    if not receipt.get('passed') or receipt.get('SFT_loading_and_smoke_gate')!='pending after S05':return False
    from q35_thinking_policy import generated_policy
    for method in ['grpo','tree_random','ucf']:
        out=path.parent/method;c=receipt['methods'][method]
        if not c['passed'] or not c['prompt_contract']['passed'] or not c.get('fresh_preupdate_base_provenance'):return False
        if sha(out/'config.json')!=c['config_sha256'] or sha(out/'rollouts_0001.jsonl.gz')!=c['rollout_sha256']:return False
        if not generated_policy(c['generated_thinking_markers'],read(out/'config.json')['model'])['passed']:return False
    return True


def evaluate_f03_once(command,out,lock):
    """One additional evaluation only; do not touch normal attempts or call SFT."""
    assert status(out/'sft/status.json')=='complete'
    assert read(RUNS/'stage0/execution_ledger.json')['user_revision_20261108']['F03_evaluation_extra_attempts']==1
    assert '--eval-only' in command and str(ROOT/'sft.py') not in command and '--resume' not in command
    where=out/'evaluation';journal=STATE/'attempts/F03_compatibility_extra_20261108.json'
    if status(where/'status.json')=='evaluation_complete' and (where/'loading_logprob_check.json').exists():
        return read(where/'loading_logprob_check.json')['passed']
    if journal.exists():return False
    if where.exists():
        archive=out/('evaluation_before_authorized_extra_'+str(int(time.time())))
        where.rename(archive)
    else:archive=None
    where.mkdir(parents=True)
    rec=dict(authorization='user 2026-11-08',command=command,extra_evaluation_attempt=1,SFT_retrained=False,
             consumes_experiment_recovery=False,started=time.time(),preserved_evaluation=str(archive) if archive else None)
    atomic(journal,rec)
    with (STATE/'F03_compatibility_extra_20261108.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
        rec['pid']=proc.pid;atomic(journal,rec)
        rec.update(returncode=proc.wait(),ended=time.time());atomic(journal,rec)
    check=where/'loading_logprob_check.json'
    return rec['returncode']==0 and status(where/'status.json')=='evaluation_complete' and check.exists() and read(check)['passed']


def prune_verified_checkpoint(where,verification):
    """Delete only this accepted checkpoint, including a previously retained target."""
    assert read(verification)['passed']
    cp=where/'checkpoint.pt'
    if not cp.exists():return
    physical=cp.resolve()
    if cp.is_symlink():
        assert physical.is_relative_to('/home/vipuser/q35-retained-checkpoints')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:assert fd.resolve()!=physical,'Checkpoint is open by a process'
                except FileNotFoundError:pass
        except (PermissionError,FileNotFoundError):pass
    rec=dict(status='planned',unix=time.time(),path=str(cp),physical_path=str(physical),bytes=cp.stat().st_size,
             sha256=sha(cp),verification_sha256=sha(verification),reason='Completed acceptance; final model and records retained')
    atomic(where/'checkpoint_pruning.json',rec)
    if cp.is_symlink():physical.unlink()
    cp.unlink();rec['status']='complete';atomic(where/'checkpoint_pruning.json',rec)


def f03_rl_smoke_gate(entry,lock,execute):
    cfg=entry['config'];assert Path(cfg['model']).resolve()==SFT.resolve()
    method=entry['method'];versioned=method+'_v22' if cfg.get('ucf_corrections') else method
    out=RUNS/'stage0/flh_f03_acceptance'/versioned
    out.mkdir(parents=True,exist_ok=True)
    receipt=out/'verification.json'
    if receipt.exists():
        saved=read(receipt)
        return saved.get('passed',False) and saved.get('ucf_corrections')==cfg.get('ucf_corrections')
    if (out/'needs_manual.json').exists():return False
    if shutil.disk_usage(RUNS).free<35*2**30:
        atomic(out/'needs_manual.json',dict(reason='Insufficient storage for F03 smoke; no GPU attempt launched'));return False
    conf=out.parent/(versioned+'_launch.json');atomic(conf,cfg)
    command=[PYTHON,'-u',str(ROOT/'train.py'),'--method',method,'--config',str(conf),'--output-dir',str(out),'--smoke']
    if not execute(command,out,'smoke_complete',lock):
        atomic(out/'needs_manual.json',dict(reason='F03 smoke computation failed after normal recovery allowance'));return False
    try:
        from transformers import AutoTokenizer
        from q35_thinking_policy import inspect_smoke
        from safetensors import safe_open
        loading=read(out/'loading_logprob_check.json');assert loading['passed'] and loading['prompt_count']==10
        checks=inspect_smoke(out,AutoTokenizer.from_pretrained(str(SFT),local_files_only=True))
        atomic(out/'smoke_checks.json',checks)
        assert checks['passed'],'SFT generated thinking markers or prompt contract failed'
        metrics=rows(out/'train_metrics.jsonl');assert len(metrics)==1 and metrics[0]['step']==1
        if cfg.get('ucf_corrections'):
            assert metrics[0]['rejection_sampling']['nonSTAY_accepted']>0
            assert metrics[0]['rejection_sampling']['forbidden_accepted']==0
            assert metrics[0]['rejection_sampling']['residual_z']<4
        assert all(math.isfinite(metrics[0][k]) for k in ['loss','kl','grad_norm','max_parameter_change'])
        for f in ['validation_0000.jsonl','validation_0001.jsonl','final_test.jsonl']:summarize(rows(out/f),4)
        shards=list((out/'final_model').glob('*.safetensors'));assert shards
        for shard in shards:
            with safe_open(shard,framework='numpy') as f:assert list(f.keys())
        atomic(receipt,dict(passed=True,smoke_only=True,source_model=str(SFT),method=method,
                           loading_check_sha256=sha(out/'loading_logprob_check.json'),thinking_checks=checks,
                           ucf_corrections=cfg.get('ucf_corrections'),
                           corrected_ucf_metrics={k:metrics[0].get(k) for k in ['candidate_coverage','mean_forks','cf_parse_mismatch_fraction','unsampled_mass_values','ucf_version','rejection_sampling']},
                           final_model_sha256={p.name:sha(p) for p in shards},test_used=False))
        prune_verified_checkpoint(out,receipt)
        # Prefix avoids collisions with unrelated per-method smoke models.
        target_out=out
        relocate_smoke_model(target_out)
        return True
    except Exception as exc:
        atomic(out/'needs_manual.json',dict(reason=repr(exc),evidence_retained=True));return False


def relocate_smoke_model(out):
    model=out/'final_model';target=Path('/home/vipuser/q35-smoke-models/flh_f03')/out.name
    if model.is_symlink():return
    target.parent.mkdir(parents=True,exist_ok=True)
    required=sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
    if shutil.disk_usage(target.parent).free<required+5*2**30:return
    if not target.exists():shutil.copytree(model,target)
    assert all(sha(p)==sha(target/p.relative_to(model)) for p in model.rglob('*') if p.is_file())
    shutil.rmtree(model);model.symlink_to(target,target_is_directory=True)
    atomic(out/'model_storage.json',dict(logical_path=str(model),physical_path=str(target),retained=True))
