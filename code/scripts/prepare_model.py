"""Download the exact base-model revision, or verify an existing local copy."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
m=json.loads((ROOT/'artifacts/model_manifest.json').read_text())
dest=ROOT/'artifacts/Qwen2.5-0.5B-Instruct'
if not a.verify_only:
 from huggingface_hub import snapshot_download
 snapshot_download(repo_id=m['repo'],revision=m['revision'],local_dir=dest,
                   allow_patterns=[f['path'] for f in m['files']])
for f in m['files']:
 path=dest/f['path']
 if not path.is_file():raise SystemExit(f'Missing model file: {path}')
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=f['sha256']:raise SystemExit(f'SHA256 mismatch: {path}')
print(f'Verified {len(m["files"])} files: {m["repo"]}@{m["revision"]}')
