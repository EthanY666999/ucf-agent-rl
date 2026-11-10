"""Verify published files against the release manifest (stdlib only)."""
import hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'RELEASE_MANIFEST.json').read_text())
for name,item in m['files'].items():
 p=root/name
 if item.get('kind')=='symlink':
  if not p.is_symlink():raise SystemExit(f'Missing symlink: {name}')
  content=os.fsencode(os.readlink(p))
 else:
  if p.is_symlink() or not p.is_file():raise SystemExit(f'Missing or unexpected symlink: {name}')
  content=p.read_bytes()
 h=hashlib.sha256(content).hexdigest()
 if h!=item['sha256'] or len(content)!=item['bytes']:raise SystemExit(f'Mismatch: {name}')
print(f'OK: {len(m["files"])} published files match SHA256/size.')
