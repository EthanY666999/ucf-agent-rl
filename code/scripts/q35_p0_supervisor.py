"""Serial F03-F15 continuation. P1/P2 and unaccepted Sokoban never start here."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from q35_p0_plan import ROOT,RUNS,PYTHON,compile_plan
from q35_run_receipt import atomic,read,sha,verify

STATE=RUNS/'queue_q35_p0'
def session(name):return subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def status(path):return read(path).get('status') if path.exists() else None
def event(**kw):
    with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),**kw),ensure_ascii=False)+'\n')


def execute(command,where,success,lock):
    """At most initial launch + one checkpoint restoration, persisted across restarts."""
    where.mkdir(parents=True,exist_ok=True) if command[2]!=str(ROOT/'sft.py') else where.parent.mkdir(parents=True,exist_ok=True)
    journal=where.parent/(where.name+'_attempts.json')
    attempts=read(journal) if journal.exists() else []
    if status(where/'status.json')==success:return True
    while len(attempts)<2:
        resume=bool(attempts)
        if resume and not (where/'checkpoint.pt').exists():return False
        record=dict(started_unix=time.time(),resume=resume,command=command)
        attempts.append(record);atomic(journal,attempts)
        # Keep the GPU lock alive in the child if this supervisor is interrupted.
        with (where.parent/(where.name+'_run.log')).open('a') as log:
            child=subprocess.Popen(command+(['--resume'] if resume else []),cwd=ROOT,
                                   stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            record['pid']=child.pid;atomic(journal,attempts)
            record['returncode']=child.wait();record['ended_unix']=time.time();atomic(journal,attempts)
        if record['returncode']==0 and status(where/'status.json')==success:return True
        event(kind='attempt_failed',directory=str(where),attempt=len(attempts),returncode=record['returncode'])
    return False


def manual(out,reason,preserve=False):
    preserved=None
    if preserve and out.exists():
        dest=out.with_name(out.name+'_failed_'+time.strftime('%Y%m%d_%H%M%S'));out.rename(dest);preserved=str(dest)
    out.mkdir(parents=True,exist_ok=True)
    atomic(out/'needs_manual.json',dict(status='待人工',reason=reason,preserved_path=preserved))
    event(kind='needs_manual',directory=str(out),reason=reason,preserved_path=preserved)


def relocate_verified_model(out,sft=False):
    """Retain every final model on the same server's larger filesystem."""
    receipt=read(out/'verification.json');assert receipt['passed']
    model=out/('sft/final_model' if sft else 'final_model')
    if model.is_symlink():return
    destination=Path('/home/vipuser/q35-final-models')/out.name
    destination.parent.mkdir(parents=True,exist_ok=True)
    required=sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
    if shutil.disk_usage(destination.parent).free<required+5*2**30:
        event(kind='model_relocation_deferred',directory=str(out),reason='remote root disk reserve');return
    # Copy and verify before removing the old copy. No model leaves the server.
    if destination.exists():
        assert all((destination/name).exists() and sha(destination/name)==h for name,h in receipt['final_model_sha256'].items())
    else:
        pending=destination.with_name(destination.name+'.copying')
        assert not pending.exists(),'Incomplete model copy requires inspection'
        shutil.copytree(model,pending)
        assert all(sha(pending/name)==h for name,h in receipt['final_model_sha256'].items())
        for path in model.rglob('*'):
            if path.is_file():assert sha(path)==sha(pending/path.relative_to(model))
        pending.rename(destination)
    shutil.rmtree(model);model.symlink_to(destination,target_is_directory=True)
    atomic(out/'model_storage.json',dict(logical_path=str(model),physical_path=str(destination),
                                       verified_sha256=receipt['final_model_sha256'],retained=True))


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    own=(STATE/'supervisor.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(STATE/'status.json',dict(status='waiting_for_F01_F02',pid=os.getpid()))
    while session('q35_admission'):time.sleep(10)
    assert read(RUNS/'stage0/acceptance.json')['passed'],'Stage0 not admitted'
    assert status(RUNS/'stage1/F01_base_eval/status.json')=='evaluation_complete'
    f02=RUNS/'stage2/F02_noSFT_GRPO4_seed42'
    assert status(f02/'status.json')=='complete' or (f02/'needs_manual.json').exists(),'F02 has no terminal receipt'
    lock=(RUNS/'gpu80.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if status(f02/'status.json')=='complete':
        try:
            verify(f02,prune_checkpoint=True)
            relocate_verified_model(f02)
        except Exception as exc:event(kind='receipt_failed',directory=str(f02),error=str(exc))
    plan=compile_plan(read(ROOT/'configs/frozenlake_q35_2b.json'),read(RUNS/'stage0/cost/result.json'))
    atomic(STATE/'plan.json',plan)
    for entry in plan['entries']:
        out=Path(entry['output'])
        if (out/'verification.json').exists() and read(out/'verification.json')['passed']:continue
        if (out/'needs_manual.json').exists():continue
        if 'F03_sft_and_evaluation' in entry['requires']:
            f03=RUNS/'stage2/F03_SFT_seed42'
            if not (f03/'verification.json').exists():
                manual(out,'Depends on unverified F03 SFT/evaluation');continue
        # Reserve room for two atomic AdamW checkpoints plus a final model.
        # No unverified checkpoints, records, or final weights are deleted here.
        if shutil.disk_usage(RUNS).free<55*2**30:
            manual(out,'Insufficient disk for atomic checkpoints; only verified completed checkpoints may be pruned');continue
        out.mkdir(parents=True,exist_ok=True)
        config=Path(entry['config_path'])
        if config.exists():assert read(config)==entry['config'],'Launch configuration changed'
        else:atomic(config,entry['config'])
        atomic(out/'hardware.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
               model_family='Qwen3.5',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               code_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST_q35.json'),environment_freeze='/data/setup_logs/qwen35_freeze_v2_20261107.txt',config_difference=entry['config_difference']))
        atomic(STATE/'status.json',dict(status='running',experiment=entry['id'],directory=str(out),pid=os.getpid()))
        if entry['method']=='sft':
            ok=execute(entry['training_command'],out/'sft','complete',lock)
            if ok:ok=execute(entry['evaluation_command'],out/'evaluation','evaluation_complete',lock)
        else:ok=execute(entry['command'],out,'complete',lock)
        if not ok:
            manual(out,'Run failed; one checkpoint recovery attempted when available',preserve=True);continue
        try:
            verify(out,sft=entry['method']=='sft',prune_checkpoint=True)
            relocate_verified_model(out,sft=entry['method']=='sft')
        except Exception as exc:
            manual(out,'Final verification failed: '+str(exc));continue
        event(kind='completed_and_verified',experiment=entry['id'],directory=str(out))
    atomic(STATE/'status.json',dict(status='FrozenLake_P0_queue_exhausted',
           complete_all=False,next='S01-S05 pending Sokoban stage0.9 acceptance',deferred=['P1','P2']))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if STATE.exists():atomic(STATE/'status.json',dict(status='supervisor_error',error=repr(exc),pid=os.getpid()))
        raise
