"""CPU-only handoff: old S04/S05 finish before a new supervisor can acquire GPU."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from q35_run_receipt import read,atomic,sha
RUNS=Path('/data/runs');STATE=RUNS/'queue_q35_hard'
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/vipuser/decision-point-q35-hard')
TOKEN='user_20261108_after_S05'


def verified(path):
    p=path/'verification.json'
    return p.exists() and read(p).get('passed') is True and read(p).get('steps')==400


def old_alive(manifest):
    p=Path('/proc')/str(manifest['old_supervisor_pid'])/'stat'
    try:return p.read_text().split()[21]==manifest['old_supervisor_start_ticks']
    except FileNotFoundError:return False


def release_markers(manifest):
    # Check every record before releasing any admission marker.
    for name in manifest['created_markers']:
        p=Path(name)
        if p.exists():assert read(p).get('handoff_token')==TOKEN
    for name,digest in manifest['authorized_manual_releases'].items():
        p=Path(name)
        if p.exists():assert sha(p)==digest, 'Manual record changed; do not override'
    for name in manifest['created_markers']:
        p=Path(name)
        if p.exists():
            assert read(p).get('handoff_token')==TOKEN
            saved=p.with_name('handoff_deferred_20261108.json')
            assert not saved.exists();p.rename(saved)
    for name in manifest['authorized_manual_releases']:
        p=Path(name)
        if p.exists():
            assert sha(p)==manifest['authorized_manual_releases'][name], 'Manual record changed; do not override'
            saved=p.with_name('needs_manual_before_authorization_20261108.json')
            assert not saved.exists();p.rename(saved)


def main():
    from q35_queue_pause import honor_user_pause
    paused = honor_user_pause('after_s05_handoff')
    if paused:
        atomic(STATE/'handoff_status.json', paused)
        return
    own=(STATE/'after_s05_handoff.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(STATE/'after_s05_handoff.json')
    atomic(STATE/'handoff_status.json',dict(status='waiting_S04_S05_unchanged',pid=os.getpid(),new_source=str(ROOT.parent)))
    while old_alive(manifest):time.sleep(20)
    # Exit of the old supervisor alone is never proof of completed experiments.
    assert verified(RUNS/'hard_queue/S04_tree_random_seed42'), 'S04 not fully verified'
    assert verified(RUNS/'hard_queue/S05_ucf_seed42'), 'S05 not fully verified'
    for name in ['S04_tree_random_seed42','S05_ucf_seed42']:
        assert read(RUNS/'hard_queue'/name/'hardware.json')['code_commit']==manifest['old_source_commit'], 'Sokoban code changed'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD,text=True).strip()==manifest['old_source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=OLD,text=True).strip()
    ready=read(STATE/'loader_revision_ready.json')
    assert ready['cpu_passed']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==ready['commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    handles=[]
    for path in [STATE/'supervisor.lock',RUNS/'gpu80.lock']:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
    release_markers(manifest)
    for f in handles:fcntl.flock(f,fcntl.LOCK_UN);f.close()
    atomic(STATE/'handoff_status.json',dict(status='starting_authorized_revision',pid=os.getpid(),source_commit=ready['commit'],S04_S05_verified=True))
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    # Run supervisor in the established continuation session, with no stale shell waiting for F02.
    command=['/data/venvs/qwen35/bin/python','-u',str(ROOT/'scripts/hard_queue_supervisor.py')]
    import shlex
    shell='cd '+shlex.quote(str(ROOT))+' && PYTHONPATH='+shlex.quote(str(ROOT))+' '+shlex.join(command)+' >> '+shlex.quote(str(STATE/'supervisor_loader_revision.log'))+' 2>&1'
    for _ in range(10):
        if subprocess.run(['tmux','has-session','-t','q35_p0_continuation'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0:break
        time.sleep(1)
    else:raise AssertionError('Old continuation session still exists; do not replace it')
    subprocess.run(['tmux','new-session','-d','-s','q35_p0_continuation',shell],check=True,env=env)
    atomic(STATE/'handoff_status.json',dict(status='launched_authorized_revision',source_commit=ready['commit'],S04_S05_verified=True,unix=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(STATE/'handoff_status.json',dict(status='needs_manual',error=repr(exc),unix=time.time()))
        raise
