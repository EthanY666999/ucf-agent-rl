"""Stop only the old dispatcher; let its current training child finish normally.

Runs outside all experiment source trees. Does not signal the training process,
change its configuration, delete artifacts, or start training without a hashed
next_dispatch.json. The final verifier is explicitly CPU-only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rpartition(') ')[2].split()
        cmd = (p / 'cmdline').read_bytes()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                    start_ticks=fields[19], cmd_sha256=hashlib.sha256(cmd).hexdigest(),
                    exit_code=int(fields[49]) if len(fields) > 49 else None)
    except FileNotFoundError:
        return None


def matching(record, observed):
    return observed is not None and str(record['start_ticks']) == observed['start_ticks']


def check_identity(record):
    p = process(record['pid'])
    assert matching(record, p), 'Process disappeared or PID was reused'
    assert p['cmd_sha256'] == record['cmd_sha256'], 'Command changed'
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    cfg = read(args.config)
    state = Path(cfg['state_dir'])
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = state / 'status.json'
    scheduler_status = Path(cfg['scheduler_status'])
    parent = cfg['supervisor']; child = cfg['child']
    assert parent['pid'] != child['pid']
    sup = check_identity(parent); train = check_identity(child)
    assert train['ppid'] == parent['pid'] and train['state'] not in ('Z', 'T', 't')
    for fd in ('1', '2'):
        assert stat.S_ISREG(os.stat('/proc/%d/fd/%s' % (child['pid'], fd)).st_mode), 'Training output must not depend on parent reading a pipe'
    for path, digest in cfg.get('protected_sha256', {}).items():
        assert sha(path) == digest, 'Protected file changed: ' + path
    if not (state / 'previous_scheduler_status.json').exists():
        atomic(state / 'previous_scheduler_status.json', read(scheduler_status))
    atomic(status, dict(status='arming_dispatch_hold', guard_pid=os.getpid(),
                        supervisor=parent, child=child, unix=time.time()))
    # Deliberately a positive individual PID: never killpg or tmux kill-session.
    os.kill(parent['pid'], signal.SIGSTOP)
    for _ in range(100):
        now = process(parent['pid'])
        if matching(parent, now) and now['state'] == 'T':
            break
        time.sleep(.01)
    else:
        raise RuntimeError('Dispatcher did not enter stopped state')
    assert check_identity(child)['state'] not in ('Z', 'T', 't'), 'Current training must remain running'
    armed = dict(status='dispatch_held_current_run_continues', experiment=cfg['experiment'],
                 guard_pid=os.getpid(), supervisor=parent, child=child,
                 run_dir=cfg['run_dir'], training_signalled=False, unix=time.time())
    atomic(status, armed); atomic(scheduler_status, armed)
    print(json.dumps(armed), flush=True)
    poll = cfg.get('poll_seconds', 15)
    while True:
        now = process(parent['pid'])
        assert matching(parent, now) and now['state'] == 'T', 'Dispatcher hold changed externally'
        current = process(child['pid'])
        if not matching(child, current) or current['state'] == 'Z':
            break
        time.sleep(poll)
    for path, digest in cfg.get('protected_sha256', {}).items():
        assert sha(path) == digest, 'Protected file changed during current run: ' + path
    atomic(state / 'child_exit_observed.json', dict(process=current, unix=time.time()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', **cfg.get('verification_env', {}))
    atomic(status, dict(armed, status='current_run_cpu_verification'))
    with (state / 'verification.log').open('a') as log:
        result = subprocess.run(cfg['verification_command'], cwd=cfg['verification_cwd'],
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Current run verification failed; old dispatch remains held'
    receipt = read(Path(cfg['run_dir']) / 'verification.json')
    assert receipt['passed'] is True
    done = dict(armed, status='current_run_verified_dispatch_paused',
                verified_unix=time.time(), verification_sha256=sha(Path(cfg['run_dir']) / 'verification.json'),
                checkpoint_pruned=False)
    atomic(status, done); atomic(scheduler_status, done)
    print(json.dumps(done), flush=True)
    # Keep the old dispatcher stopped and retaining its existing locks until a
    # reviewed replacement is ready. No old queued item can be launched here.
    while not (state / 'next_dispatch.json').exists():
        time.sleep(poll)
    nxt = read(state / 'next_dispatch.json')
    assert nxt['authorized'] is True and nxt['files_sha256']
    for path, digest in nxt['files_sha256'].items():
        assert sha(path) == digest, 'New dispatcher admission hash mismatch: ' + path
    assert check_identity(parent)['state'] == 'T'
    current = process(child['pid'])
    assert not matching(child, current) or current['state'] == 'Z'
    # The superseded scheduler is stopped and the training child has exited.
    # KILL avoids briefly resuming it into S10. This is not a process-group signal.
    os.kill(parent['pid'], signal.SIGKILL)
    for _ in range(100):
        obs = process(parent['pid'])
        if not matching(parent, obs) or obs['state'] == 'Z':
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Superseded dispatcher did not exit')
    with (state / 'next_dispatch.log').open('a') as log:
        new = subprocess.Popen(nxt['command'], cwd=nxt['cwd'],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True,
                               env=dict(os.environ, **nxt.get('env', {})))
    atomic(status, dict(done, status='handed_to_v3_dispatcher', new_dispatch_pid=new.pid,
                        new_command=nxt['command'], handed_off_unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not resume the obsolete queue automatically on any guard error.
        args = argparse.ArgumentParser(); args.add_argument('--config', type=Path, required=True)
        parsed = args.parse_args()
        cfg = read(parsed.config)
        atomic(Path(cfg['state_dir']) / 'error.json', dict(error=repr(exc), unix=time.time(),
               old_queue_resume_authorized=False, training_signalled=False))
        raise
