"""Fail-closed, serial UCF-v2.2-ff admission; no GPU before all offline checks."""
import json, math, os, shutil, subprocess, time
from pathlib import Path
from q35_p0_plan import ROOT,RUNS,PYTHON
from q35_run_receipt import read,atomic,sha
STATE=RUNS/'queue_q35_hard/ucf_v22_20261109'

def corrected_cpu_ready():
    names=['cpu_functional.json','htsim_first_fence.json','prefix_causality.json']
    if not all((STATE/name).exists() and read(STATE/name).get('passed') is True for name in names):return False
    return read(STATE/'cpu_functional.json').get('source_manifest_sha256')==sha(ROOT/'SOURCE_MANIFEST_q35.json')

def offline_passed(report):
    return (report.get('passed') is True and report.get('mismatch_threshold')==.02
            and report.get('mismatch_fraction',1.)<=.02 and report.get('training_only') is True
            and report.get('response_protocol')=='first_complete_action_block'
            and report.get('source_experiment')=='S13' and report.get('complete_steps')==400)

def gpu_admission(entry, lock):
    STATE.mkdir(parents=True,exist_ok=True)
    s13=RUNS/'hard_queue/S13_grpo_seed42'
    assert read(s13/'verification.json').get('passed') is True
    # User authorized a new protocol. F02 remains historical and unpaired.
    # Only actual new-protocol S13 training may satisfy this unchanged 2% gate.
    s13_report=STATE/'S13_offline.json'
    if not s13_report.exists():
        with (STATE/'S13_offline.log').open('a') as log:
            rc=subprocess.call([PYTHON,str(ROOT/'scripts/q35_response_protocol_audit.py'),'--run',str(s13),'--output',str(s13_report),'--new-protocol'],stdout=log,stderr=subprocess.STDOUT)
        if rc:atomic(STATE/'admission.json',dict(passed=False,reason='S13 offline audit failed',returncode=rc));return False
    reasons=[]
    for name in ['S13_offline.json']:
        path=STATE/name
        if not path.exists() or not offline_passed(read(path)):reasons.append(name+' mismatch gate not passed')
    for name in ['cpu_functional.json','htsim_first_fence.json','prefix_causality.json']:
        path=STATE/name
        if not path.exists() or read(path).get('passed') is not True:reasons.append(name+' not passed')
    if not corrected_cpu_ready():reasons.append('CPU receipts do not match current source manifest')
    if reasons:
        atomic(STATE/'admission.json',dict(passed=False,status='needs_manual',reasons=reasons,GPU_started=False));return False
    acceptance=STATE/'gpu_acceptance.json'
    if acceptance.exists():
        receipt=read(acceptance)
        assert receipt['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST_q35.json')
        return receipt.get('passed') is True
    journal=STATE/'gpu_smoke_attempt.json'
    if journal.exists():return False  # Preserve uncertain/failed attempt; never duplicate.
    if shutil.disk_usage(RUNS).free<35*2**30:
        atomic(STATE/'admission.json',dict(passed=False,reason='Insufficient smoke storage'));return False
    out=STATE/'gpu_smoke';conf=STATE/'gpu_smoke_config.json'
    # Use S15 production cap/sampling even though --smoke reduces batch to four.
    cfg=dict(entry['config']);assert cfg['fork_point']=='first_fence'
    atomic(conf,cfg)
    command=[PYTHON,'-u',str(ROOT/'ucf/run_ucf.py'),'--config',str(conf),'--output-dir',str(out),'--fork_point','first_fence','--smoke']
    record=dict(started=time.time(),command=command,steps=1,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST_q35.json'),GPU_started=True)
    atomic(journal,record)
    with (STATE/'gpu_smoke.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
        record['pid']=proc.pid;atomic(journal,record);record['returncode']=proc.wait();record['ended']=time.time();atomic(journal,record)
    if record['returncode']!=0:return False
    metrics=[json.loads(x) for x in (out/'train_metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==1 and metrics[0]['step']==1
    from transformers import AutoTokenizer
    from q35_thinking_policy import inspect_smoke
    thinking=inspect_smoke(out,AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True))
    atomic(out/'thinking_checks.json',thinking)
    assert thinking['passed'], 'Prompt/generated-marker gate failed'
    m=metrics[0];assert all(x==0 for x in m['unsampled_mass_values'])
    assert m['rejection_sampling']['nonSTAY_accepted']>0
    assert m['rejection_sampling']['forbidden_accepted']==0 and m['rejection_sampling']['residual_z']<4
    finite=all(math.isfinite(m[k]) for k in ['loss','kl','grad_norm','max_parameter_change'])
    receipt=dict(passed=finite and m['max_parameter_change']>0 and read(out/'status.json')['status']=='smoke_complete',
      source_manifest_sha256=record['source_manifest_sha256'],thinking_checks=thinking,coverage=m['first_fence_coverage'],
      candidate_coverage=m['candidate_coverage'],mean_forks=m['mean_forks'],expected_mean_forks=m['fork_budget']['expected_forks'],cf_parse_mismatch=m['cf_parse_mismatch_fraction'],
      finite=finite,nonzero_update=m['max_parameter_change']>0,production_reply_cap=cfg['max_response_tokens'],
      rejection_sampling=m['rejection_sampling'],smoke_only=True,initializes_no_experiment=True,record_sha256={'train_metrics.jsonl':sha(out/'train_metrics.jsonl')})
    atomic(acceptance,receipt)
    # Do not delete any failed acceptance evidence or use its weights for S15.
    if receipt['passed'] and (out/'checkpoint.pt').exists():
        cp=out/'checkpoint.pt';atomic(out/'checkpoint_pruning.json',dict(bytes=cp.stat().st_size,reason='one-step validation and finite nonzero update verified; final smoke model retained'));cp.unlink()
    return receipt['passed']
