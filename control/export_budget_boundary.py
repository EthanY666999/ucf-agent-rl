"""Build a lightweight, lossless evidence bundle from a *completed* run boundary.

Read only the named fixed validation and completed-step prefix. Never opens or
copies model weights, checkpoint data, raw training leaves or a test split.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(run, step, output):
    receipt_path = run / f'stage_{step:04}.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['passed'] and receipt['steps'] == step and not receipt['final_test_used']
    first = receipt['initial_step'] + 1
    metrics_path = run / 'train_metrics.jsonl'
    with metrics_path.open('rb') as f:
        lines = [next(f) for _ in range(first, step + 1)]
    assert [json.loads(line)['step'] for line in lines] == list(range(first, step+1))
    metrics = b''.join(lines)
    assert digest(metrics) == receipt['metrics_sha256']
    validation = run / f'validation_{step:04}.json'
    records = validation.with_suffix('.jsonl')
    assert digest(validation.read_bytes()) == receipt['validation_sha256'][str(step)]
    assert sum(1 for l in records.read_bytes().splitlines() if l.strip()) == 400
    manifest = {}
    tmp = output.with_name(output.name+'.tmp')
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tmp.open('wb') as dest, gzip.GzipFile(fileobj=dest,filename='',mode='wb',mtime=0) as gz, tarfile.open(fileobj=gz,mode='w|') as tar:
            def add(name,raw):
                info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
                tar.addfile(info,io.BytesIO(raw))
            def archive(source, data=None, compress=False):
                assert source.is_file() and not source.is_symlink()
                data=source.read_bytes() if data is None else data
                rel=source.relative_to(run).as_posix()
                stored=gzip.compress(data,compresslevel=6,mtime=0) if compress else data
                name=rel+('.gz' if compress else '')
                assert len(stored)<99_000_000 and name not in manifest
                manifest[name]=dict(source=str(source),uncompressed_bytes=len(data),uncompressed_sha256=digest(data),
                                    archived_bytes=len(stored),archived_sha256=digest(stored))
                add(name,stored)
            for name in ('config.json','continuation_config.json','initialization.json','hardware.json',
                         'prompt_contract.json','numerical_runtime.json'):
                archive(run/name)
            archive(receipt_path);archive(metrics_path,metrics);archive(validation);archive(records,compress=True)
            if (run/'ucf_logs').is_dir():
                for s in range(first,step+1):
                    for pattern in ('budget_{:04}.json','controller_{:04}.json','select_{:04}.jsonl'):
                        p=run/'ucf_logs'/pattern.format(s)
                        archive(p,compress=pattern.endswith('.jsonl'))
            details=dict(run=str(run),step=step,initial_step=first-1,final_test_used=False,
                         models_checkpoints_included=False,training_rollouts_retained_on_server=True,files=manifest)
            add('boundary_inventory.json',(json.dumps(details,indent=2)+'\n').encode())
        # Re-open the actual archive before replacing any completed bundle.
        with tarfile.open(tmp,'r:gz') as tf:
            assert set(tf.getnames())==set(manifest)|{'boundary_inventory.json'}
            for name,entry in manifest.items():
                assert digest(tf.extractfile(name).read())==entry['archived_sha256']
        os.replace(tmp,output)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(files=len(manifest)+1,bytes=output.stat().st_size,sha256=digest(output.read_bytes()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.run,a.step,a.output)))
