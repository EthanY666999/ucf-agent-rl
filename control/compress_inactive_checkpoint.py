"""Lossless archive of the inactive default-kernel diagnostic checkpoint only.

Preparation is non-destructive. Commit requires a published manifest and verifies
the decompressed stream again. The active scoped experiment is never touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OLD = Path('/home/vipuser/ucf-v3-runs-20261110/V3_S_grpo_nosft_seed42')
SOURCE = OLD / 'checkpoint.pt'
STAGING = Path('/data/runs/ucf_v3_storage_20261110/default_grpo25_checkpoint.pt.zst')
FINAL = OLD / 'checkpoint.pt.zst'
EXPECTED = 'd3fdd5fd96aba34bbf9c2b0a61fb87c25424c8c70170e2a6ec44a08635517e3d'
EXPECTED_BYTES = 22582328681

def write(path, obj):
    tmp = path.with_name(path.name + '.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def verify_archive(path):
    p = subprocess.Popen(['zstd', '-dc', '--', str(path)], stdout=subprocess.PIPE)
    h = hashlib.sha256(); size = 0
    for b in iter(lambda: p.stdout.read(8 * 1024**2), b''):
        h.update(b); size += len(b)
    assert p.wait() == 0
    assert size == EXPECTED_BYTES and h.hexdigest() == EXPECTED

def inactive():
    s = json.loads((OLD / 'status.json').read_text())
    assert s['status'] == 'paused_checkpointed' and s['step'] == 25
    assert json.loads((OLD / 'stage_0025.json').read_text())['passed']
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if any(Path(os.fsdecode(a)).name in {'train.py', 'train_scoped.py'} for a in args):
            assert str(OLD).encode() not in args, 'Old training process is active'

def prepare():
    inactive()
    assert SOURCE.is_file() and not SOURCE.is_symlink()
    assert SOURCE.stat().st_size == EXPECTED_BYTES and digest(SOURCE) == EXPECTED
    STAGING.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(STAGING.parent).free > EXPECTED_BYTES + 8 * 1024**3
    assert not STAGING.exists(), 'Archive already exists; inspect it before another attempt'
    tmp = STAGING.with_suffix('.zst.tmp')
    assert not tmp.exists()
    before = {m: shutil.disk_usage(m).free for m in ['/home/vipuser', '/data']}
    subprocess.run(['nice', '-n', '19', 'zstd', '-T1', '-3', '-q', '-o', str(tmp), '--', str(SOURCE)], check=True)
    verify_archive(tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(STAGING)
    record = dict(status='verified_archive_original_retained', original=str(SOURCE), original_bytes=EXPECTED_BYTES,
                  original_sha256=EXPECTED, staging_archive=str(STAGING), final_archive=str(FINAL),
                  archive_bytes=STAGING.stat().st_size, archive_sha256=digest(STAGING),
                  decompressed_sha256=EXPECTED, lossless=True, model_optimizer_rng_preserved=True,
                  active_scoped_run_untouched=True, before_free_bytes=before, unix=time.time(),
                  reason='Inactive default-numerics GRPO25 is retained for reproducibility, not in the current paired queue')
    write(STAGING.parent / 'compression_manifest.json', record)
    print(json.dumps(record), flush=True)

def commit(manifest_sha, published_commit):
    inactive()
    mf = STAGING.parent / 'compression_manifest.json'
    assert digest(mf) == manifest_sha
    record = json.loads(mf.read_text())
    assert len(published_commit) == 40 and all(c in '0123456789abcdef' for c in published_commit)
    assert record['archive_sha256'] == digest(STAGING)
    verify_archive(STAGING)
    assert SOURCE.stat().st_size == EXPECTED_BYTES and digest(SOURCE) == EXPECTED
    assert not FINAL.exists()
    # All bytes, including optimizer/RNG, already exist in a verified archive.
    # Original metadata and evaluation/source/rollout files stay in place.
    SOURCE.unlink()
    record.update(status='original_replaced_by_verified_archive', published_commit=published_commit,
                  published_manifest_sha256=manifest_sha, source_unlinked_unix=time.time())
    write(STAGING.parent / 'compression_receipt.json', record)
    assert shutil.disk_usage(OLD).free > record['archive_bytes'] + 32 * 1024**3
    tmp = FINAL.with_suffix('.zst.tmp')
    assert not tmp.exists()
    shutil.copyfile(STAGING, tmp)
    assert digest(tmp) == record['archive_sha256']
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(FINAL)
    STAGING.unlink()
    record.update(status='lossless_archive_complete', actual_archive=str(FINAL),
                  released_bytes=EXPECTED_BYTES-record['archive_bytes'],
                  after_free_bytes={m: shutil.disk_usage(m).free for m in ['/home/vipuser', '/data']},
                  restored_plaintext_required_before_resume=True, completed_unix=time.time())
    write(OLD / 'checkpoint_archive.json', record)
    write(STAGING.parent / 'compression_receipt.json', record)
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'commit'])
    p.add_argument('--manifest-sha256'); p.add_argument('--published-commit'); a = p.parse_args()
    prepare() if a.mode == 'prepare' else commit(a.manifest_sha256, a.published_commit)
