"""Finish the existing UCF50, run exactly its matched GRPO50, then review on CPU."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/data/runs/ucf_v3_budget10240_20261110')
WORK = ROOT / 'paired50_handoff'
QUEUE = ROOT / 'training'
ARCHIVE = 'uniform_ucf_step0030.pt.zst'
PARKED = Path('/home/vipuser/ucf-v3-anchors-20261111') / ARCHIVE
GIB = 2**30
CPU = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
EXPECTED_PLAN = 'f3929f9ac2c73fad3432565b09efb566894401c965c9095e3f346a9611c3d6e0'


def read(path):
    return json.loads(path.read_bytes())


def write(path, record):
    temp = path.with_name(path.name + '.writing')
    with temp.open('w') as file:
        json.dump(record, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def checkpoint_step(python, path):
    code = 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",mmap=True,weights_only=False)["step"])'
    return int(subprocess.check_output([python, '-c', code, str(path)], env=CPU, text=True))


def capture(python, run):
    source = run / 'checkpoint.pt'
    receipt = WORK / 'anchor.json'
    if receipt.exists():
        record = read(receipt)
        saved = PARKED if PARKED.exists() else WORK / ARCHIVE
        assert saved.stat().st_size == record['archive_bytes'] and sha(saved) == record['archive_sha256']
        return record
    stat = source.stat()
    assert checkpoint_step(python, source) == 30
    # Only the system partition is being written by the running trainer.
    assert shutil.disk_usage(WORK).free > stat.st_size + 10 * GIB
    dst = WORK / ARCHIVE
    temp = dst.with_suffix('.zst.writing')
    assert not temp.exists() and not dst.exists()
    subprocess.run(['ionice','-c','3','nice','-n','15','zstd','-q','-1','-T1',
                    '-o',str(temp),'--',str(source)], check=True)
    digest = sha(source)
    assert (source.stat().st_ino, source.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
    with subprocess.Popen(['zstd','-qdc','--',str(temp)], stdout=subprocess.PIPE) as unpack:
        checked = hashlib.sha256()
        size = 0
        for block in iter(lambda: unpack.stdout.read(4 * 1024**2), b''):
            checked.update(block)
            size += len(block)
        assert unpack.wait() == 0 and size == stat.st_size and checked.hexdigest() == digest
    with temp.open('rb') as file:
        os.fsync(file.fileno())
    temp.replace(dst)
    record = dict(step=30, source=str(source), source_bytes=size, source_sha256=digest,
                  archive_bytes=dst.stat().st_size, archive_sha256=sha(dst), staged_path=str(dst),
                  final_path=str(PARKED), decompressed_sha256=digest,
                  source_method='uniform UCF, seed42, 10240 retained tokens', model_optimizer_rng=True,
                  comparison='same uniform UCF trajectory through step30; fork for updates 31-50',
                  captured_unix=time.time())
    write(receipt, record)
    return record


def park(record):
    # The old UCF writer has completed both saves and its manager has released its locks.
    # Keep the anchor on the idle system partition while GRPO uses the data partition.
    src = WORK / ARCHIVE
    PARKED.parent.mkdir(exist_ok=True)
    if not PARKED.exists():
        assert shutil.disk_usage(PARKED.parent).free > record['archive_bytes'] + 10 * GIB
        assert src.stat().st_size == record['archive_bytes'] and sha(src) == record['archive_sha256']
        temp = PARKED.with_suffix('.zst.writing')
        assert not temp.exists()
        shutil.copyfile(src, temp)
        assert sha(temp) == record['archive_sha256']
        with temp.open('rb') as file:
            os.fsync(file.fileno())
        temp.replace(PARKED)
    assert PARKED.stat().st_size == record['archive_bytes'] and sha(PARKED) == record['archive_sha256']
    if src.exists():
        assert src.stat().st_size == record['archive_bytes'] and sha(src) == record['archive_sha256']
        src.unlink()  # Remove the verified staging copy only.
    write(WORK / 'parked.json', dict(path=str(PARKED),sha256=record['archive_sha256'],
          verified_staging_copy_removed=True,free_gib={p:shutil.disk_usage(p).free/GIB for p in ['/home','/data']},
          unix=time.time()))


def main():
    WORK.mkdir(exist_ok=True)
    lock = (WORK / 'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    planpath = ROOT / 'plan.json'
    assert sha(planpath) == EXPECTED_PLAN
    plan = read(planpath)
    ucf, grpo = plan['entries'][2:4]
    assert [(e['method'],e['stop_step']) for e in [ucf,grpo]] == [('ucf',50),('grpo',50)]
    runs = [Path(e['output']) for e in [ucf,grpo]]
    for run in runs:
        assert read(run/'stage_0025.json')['passed']
    newplan = dict(plan, state=str(WORK/'matched_grpo'), entries=[grpo])
    write(WORK/'grpo50_only.json',newplan)
    rule = dict(owner='paired50_handoff', original_plan_sha256=EXPECTED_PLAN,
                continue_after_ucf='same matched GRPO configuration through step 50 only',
                successor=str(WORK/'grpo50_only.json'), stop_after_pair=True,
                candidate_margin_percentage_points=-2, rescue_windows=[[17,25],[42,50]],
                rescue_denominator='all four-original-failure training questions including those with zero selected forks',
                uncertainty='report question-paired bootstrap; this is an exploratory resource rule, not a statistical noninferiority claim')
    write(WORK/'decision.json',rule)
    hold = QUEUE/'hold_after_current.json'
    if hold.exists():
        assert read(hold) == rule
    else:
        live = read(QUEUE/'status.json')
        assert live['experiment']==ucf['id'] and live['stage']==50 and live['child_alive']
        write(hold,rule)
    write(WORK/'status.json',dict(phase='preserving_step30',unix=time.time()))
    record = capture(plan['python'],runs[0])
    write(WORK/'status.json',dict(phase='watching_active_ucf50',unix=time.time()))
    while not (QUEUE/'boundary_held.json').exists():
        if any((QUEUE/'failures').iterdir()):
            raise RuntimeError('Active manager recorded a failure')
        time.sleep(20)
    assert read(QUEUE/'boundary_held.json')['next_entry']==grpo
    assert read(runs[0]/'stage_0050.json')['passed']
    old_manager_pid=read(ROOT/'training_launch.json')['pid']
    while Path(f'/proc/{old_manager_pid}/cmdline').exists():
        try:
            if str(plan['manager_script']).encode() not in Path(f'/proc/{old_manager_pid}/cmdline').read_bytes():
                break
        except FileNotFoundError:
            break
        time.sleep(1)
    write(WORK/'status.json',dict(phase='moving_verified_anchor_to_idle_mount',unix=time.time()))
    park(record)
    assert shutil.disk_usage(runs[1]).free >= 32 * GIB
    write(WORK/'status.json',dict(phase='matched_grpo50',unix=time.time()))
    subprocess.run([plan['python'],'-u',plan['manager_script'],str(WORK/'grpo50_only.json')],check=True)
    assert read(runs[1]/'stage_0050.json')['passed']
    lockpath=ROOT/'reports/review_0050.lock'
    with lockpath.open('a') as review_lock:
        fcntl.flock(review_lock,fcntl.LOCK_EX)
        export=ROOT/'reports/paired_export_0050.json'
        if not export.exists():
            subprocess.run(['bash',str(ROOT/'review_budget_pair.sh'),str(ROOT),'50'],check=True,env=CPU)
        assert read(export)['passed']
    write(WORK/'status.json',dict(phase='both_50_verified_and_held',archives_verified=True,unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if WORK.is_dir():
            write(WORK/'handoff_error.json',{'error':repr(error),'unix':time.time()})
        raise
