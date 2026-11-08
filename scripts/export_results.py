"""Export an allowlisted light evidence bundle; never weights or raw rollouts."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ALLOWED_NAMES={'config.json','launch_config.json','train_metrics.jsonl','metrics.jsonl','status.json',
               'sft_acceptance.json','source_versions.jsonl','branch_events.jsonl',
               'release_receipt.json','release_manifest.json','additional_metrics.json','verification.json','run_report.json'}
def allowed(rel):
    if 'source_snapshots' in rel.parts:return rel.suffix=='.py'
    if 'ucf_logs' in rel.parts:
        return not rel.name.startswith('raw_leaves') and rel.suffix in {'.json','.jsonl'}
    return rel.name in ALLOWED_NAMES or ((rel.name.startswith('validation_') or rel.name.startswith('final_test') or 'supplemental500' in rel.parts) and rel.suffix in {'.json','.jsonl'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    run=a.run.resolve();files=[f for f in sorted(run.rglob('*')) if f.is_file() and allowed(f.relative_to(run))]
    if not files:raise SystemExit('No evidence found')
    manifest={str(f.relative_to(run)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Partition large light evidence bundles before publication; no automatic upload.
    with tarfile.open(a.output,'w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(run)),recursive=False)
        data=(json.dumps(dict(files=manifest,excluded='weights, checkpoints, raw rollout leaves'),indent=2)+'\n').encode()
        info=tarfile.TarInfo('evidence_manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    size=a.output.stat().st_size
    print(json.dumps(dict(bundle=str(a.output),bytes=size,files=len(files),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))
    if size>=95*1024**2:raise SystemExit('Bundle exceeds release limit; retain locally and split before submitting. Nothing uploaded.')


if __name__=='__main__':main()
