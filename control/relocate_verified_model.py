"""Remote-only verified relocation; preserve the existing model path by symlink."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def main():
    plan_path=Path(sys.argv[1]);plan=json.loads(plan_path.read_text())
    publication_commit=sys.argv[2]
    assert len(publication_commit)==40 and all(c in '0123456789abcdef' for c in publication_commit)
    source,dest=Path(plan['source']),Path(plan['destination'])
    assert str(source)=='/home/vipuser/q35-final-models/S07_starpo_s_seed42/model.safetensors'
    assert str(dest)=='/data/runs/retained_models/S07_starpo_s_seed42/model.safetensors'
    assert source.is_file() and not source.is_symlink()
    before={p:shutil.disk_usage(p).free for p in ['/home/vipuser','/data']}
    assert before['/data']>plan['bytes']+15*2**30
    stat=source.stat()
    assert stat.st_size==plan['bytes'] and sha(source)==plan['sha256']
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(), 'Do not overwrite a retained model'
    tmp=dest.with_name(dest.name+f'.partial-{os.getpid()}')
    with source.open('rb') as f,tmp.open('xb') as g:
        shutil.copyfileobj(f,g,8*1024*1024);g.flush();os.fsync(g.fileno())
    assert tmp.stat().st_size==plan['bytes'] and sha(tmp)==plan['sha256']
    assert source.stat().st_mtime_ns==stat.st_mtime_ns and source.stat().st_size==stat.st_size
    shutil.copystat(source,tmp);tmp.replace(dest)
    link=source.with_name(source.name+f'.verified-link-{os.getpid()}')
    link.symlink_to(dest);link.replace(source)
    assert source.resolve()==dest and source.stat().st_size==plan['bytes']
    after={p:shutil.disk_usage(p).free for p in before}
    result=dict(passed=True,unix=time.time(),source=str(source),destination=str(dest),
        bytes=plan['bytes'],sha256=plan['sha256'],plan_sha256=sha(plan_path),
        published_plan_commit=publication_commit,source_now_symlink=True,
        model_retained=True,evidence_deleted=False,mac_copy=False,
        disk_free_bytes_before=before,disk_free_bytes_after=after,
        total_storage_reduced=False)
    receipt=plan_path.with_name('S07_model_relocation_receipt_20261110.json')
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
