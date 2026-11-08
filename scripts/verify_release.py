"""Check immutable code/config/data hashes and publication hygiene using CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'RELEASE_MANIFEST.json'
TREES=('code','configs','scripts','tests','data')
DEPENDENCIES=('requirements-cpu.txt','requirements-gpu.txt')


def files():
    for name in DEPENDENCIES:
        yield ROOT/name
    for tree in TREES:
        for p in sorted((ROOT/tree).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
            # Production train/SFT create the execution lock and receipts here.
            # Runtime files are not part of the frozen source/data contract.
            if tree=='code' and p.is_relative_to(ROOT/'code/artifacts'):continue
            yield p


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory():
    result={}
    for p in files():
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix in {'.pt','.bin','.safetensors','.onnx','.log'} or p.name.startswith('rollouts'):
            raise ValueError('Forbidden release payload: '+rel)
        if p.stat().st_size>=95*1024**2:raise ValueError('Oversized GitHub payload: '+rel)
        if p.is_symlink():raise ValueError('Release must be self contained: '+rel)
        result[rel]=dict(sha256=digest(p),bytes=p.stat().st_size)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');a=p.parse_args()
    current=inventory()
    if a.build:
        MANIFEST.write_text(json.dumps(dict(schema_version=1,algorithm_origin_commit='e32ac58b8a857eb98023ec1f79bbe0cf855c95ac',
            scope='root dependency pins, code, configs, scripts, tests and published data; no weights or internal logs',files=current),indent=2)+'\n')
        print(f'Created manifest for {len(current)} files');return
    expected=json.loads(MANIFEST.read_text())['files']
    changed=[k for k in sorted(set(expected)|set(current)) if expected.get(k)!=current.get(k)]
    if changed:raise SystemExit('Release hash mismatch. Stop and report: '+', '.join(changed))
    print(json.dumps(dict(verified_files=len(current),bytes=sum(v['bytes'] for v in current.values()),GPU_used=False)))


if __name__=='__main__':main()
