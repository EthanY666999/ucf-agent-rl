#!/usr/bin/env python3
"""Expand the pinned text games and compact task metadata; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data/alfworld')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.data_dir/'source_manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        if hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest()!=expected['sha256']:
            raise RuntimeError('ALFWorld archive hash mismatch: '+name)
    with zipfile.ZipFile(a.data_dir/'text_games.zip') as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise RuntimeError('Unsafe archive member')
        z.extractall(a.output_dir)
    metadata=json.loads((a.data_dir/'task_metadata.json').read_text())
    for name, record in metadata.items():
        destination=a.output_dir/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(record)+'\n')
    print(json.dumps({'games_root':str(a.output_dir/'json_2.1.1'),'metadata_records':len(metadata)}))

if __name__=='__main__':main()
