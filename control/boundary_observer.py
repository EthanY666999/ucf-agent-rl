"""Observe an admission-revoked dispatcher and hand off only after S07 completes.

No process signals. Revocation is performed separately and recorded. The old
dispatcher finishes its current child and receipt, then its next admission fails.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from boundary_guard import atomic, read, sha, process, matching


def main():
    cfg = read(sys.argv[1]); state = Path(cfg['state_dir'])
    own = (state / 'observer.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = cfg['supervisor']; run = Path(cfg['run_dir'])
    ready = Path(cfg['old_ready'])
    while True:
        revoked = read(ready)
        assert revoked['passed'] is False and revoked['revoked_by'] == 'user_ucf_v3_20261110'
        for path, digest in cfg['protected_sha256'].items():
            assert sha(path) == digest, 'Active source/config changed: ' + path
        obs = process(parent['pid'])
        active = read(run / 'status.json')
        atomic(state / 'status.json', dict(status='old_pending_admission_revoked_current_run_continues',
               experiment='S07', training_status=active['status'], step=active.get('step'),
               old_dispatcher_alive=matching(parent, obs) and obs['state'] != 'Z',
               observer_pid=os.getpid(), training_signalled=False, unix=time.time()))
        if not matching(parent, obs) or obs['state'] == 'Z':
            break
        time.sleep(15)
    # The current child and any verifier must have relinquished the shared locks.
    handles = []
    for path in cfg['handoff_locks']:
        fd = Path(path).open('a'); fcntl.flock(fd, fcntl.LOCK_EX); handles.append(fd)
    assert read(run / 'status.json')['status'] == 'complete', 'S07 needs recovery; preserve its checkpoint'
    receipt = run / 'verification.json'
    if not receipt.exists() or not read(receipt).get('passed'):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', **cfg.get('verification_env', {}))
        with (state / 'verification.log').open('a') as log:
            subprocess.run(cfg['verification_command'], cwd=cfg['verification_cwd'], env=env,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(receipt)['passed'] is True
    atomic(state / 'status.json', dict(status='S07_verified_waiting_for_v3_readiness',
           verification_sha256=sha(receipt), observer_pid=os.getpid(), unix=time.time()))
    while not (state / 'next_dispatch.json').exists():
        time.sleep(10)
    nxt = read(state / 'next_dispatch.json')
    assert nxt['authorized'] is True and nxt['files_sha256']
    for path, digest in nxt['files_sha256'].items():
        assert sha(path) == digest, 'New admission hash mismatch: ' + path
    assert not (state / 'handoff_launched.json').exists(), 'Replacement was already launched'
    env = dict(os.environ, **nxt.get('env', {}),
               UCF_INHERITED_QUEUE_LOCK_FD=str(handles[0].fileno()),
               UCF_INHERITED_GPU_LOCK_FD=str(handles[1].fileno()))
    with (state / 'next_dispatch.log').open('a') as log:
        new = subprocess.Popen(nxt['command'], cwd=nxt['cwd'], env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, pass_fds=tuple(h.fileno() for h in handles))
    event = dict(status='handed_to_v3', pid=new.pid, command=nxt['command'], unix=time.time())
    atomic(state / 'handoff_launched.json', event); atomic(state / 'status.json', event)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        cfg = read(sys.argv[1])
        atomic(Path(cfg['state_dir']) / 'observer_error.json', dict(error=repr(exc), unix=time.time(),
               automatic_old_queue_resume=False))
        raise
