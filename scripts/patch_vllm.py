"""Apply only the audited vLLM 0.19.1 allowed-token row-swap fix (CPU)."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = '''                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],'''
NEW = '''                self.allowed_token_ids_mask_cpu_tensor[i2].clone(),
                self.allowed_token_ids_mask_cpu_tensor[i1].clone(),'''


def patched(text):
    if NEW in text and OLD not in text:
        return text
    if text.count(OLD) != 1:
        raise ValueError('Unknown source: expected exactly one audited swap; refusing to guess')
    return text.replace(OLD, NEW, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    dist = importlib.metadata.distribution('vllm')
    if dist.version != '0.19.1':
        raise SystemExit('This patch is pinned to vllm==0.19.1')
    file = Path(dist.locate_file('vllm/v1/worker/gpu_input_batch.py'))
    manifest = json.loads((ROOT/'code/patches/manifest.json').read_text())
    before = file.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    if digest == manifest['after_sha256']:
        print('Already patched; SHA256 verified.'); return
    if digest != manifest['before_sha256']:
        raise SystemExit('Installed source hash differs from audited upstream. Stop and report.')
    if not a.apply:
        raise SystemExit('Patch is required. Run this script with --apply before GPU smoke.')
    after = patched(before.decode()).encode()
    assert hashlib.sha256(after).hexdigest() == manifest['after_sha256']
    file.with_suffix('.py.ucf-original').write_bytes(before)
    temporary = file.with_suffix('.py.ucf-tmp')
    temporary.write_bytes(after)
    temporary.replace(file)
    print('Applied audited row-swap fix; SHA256 verified. No GPU used.')


if __name__ == '__main__':
    main()
