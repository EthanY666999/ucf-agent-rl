"""Losslessly store an inactive checkpoint in independently verifiable zstd parts."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from lossless_weight_archive import Codec, delta_bytes, original_bytes


BLOCK = 8 * 1024**2
PART = 1024**3


def transformed(data, reverse=False):
    n = len(data) // 4 * 4
    words = np.frombuffer(data[:n], dtype=np.uint8)
    matrix = words.reshape(4, -1) if reverse else words.reshape(-1, 4)
    return matrix.T.tobytes() + data[n:]


def zstd_input(path):
    return subprocess.Popen(["zstd", "-qdc", "--", str(path)], stdout=subprocess.PIPE)


@contextmanager
def reference_input(ref):
    if not ref:
        yield None
        return
    digest = hashlib.sha256()
    with zstd_input(ref["path"]) as reader:
        def take(size):
            data = reader.stdout.read(size)
            assert len(data) == size
            digest.update(data)
            return data
        take(ref["skip"])
        yield take
        for block in iter(lambda:reader.stdout.read(BLOCK), b""):
            digest.update(block)
        assert reader.wait() == 0 and digest.hexdigest() == ref["raw_sha256"]


def block_bytes(data, ref_read, reverse=False):
    if ref_read is None:
        return transformed(data, reverse=reverse)
    reference = ref_read(len(data))
    n = len(data) // 4 * 4
    codec = original_bytes if reverse else delta_bytes
    return codec(data[:n], reference[:n]) + data[n:]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def verify_parts(index, dest, ref_read):
    digest = hashlib.sha256()
    total = 0
    for part in index["parts"]:
        path = Path(part["path"])
        assert path.stat().st_size == part["packed_bytes"] and file_hash(path) == part["packed_sha256"]
        with zstd_input(path) as reader:
            remain = part["source_bytes"]
            while remain:
                size = min(BLOCK, remain)
                data = reader.stdout.read(size)
                assert len(data) == size
                raw = block_bytes(data, ref_read, reverse=True)
                digest.update(raw)
                if dest:
                    dest.write(raw)
                remain -= size
                total += size
            assert reader.stdout.read(1) == b"" and reader.wait() == 0
    assert total == index["source_bytes"] and digest.hexdigest() == index["source_sha256"]


def verify(index, dest=None):
    with reference_input(index.get("reference")) as ref_read:
        verify_parts(index, dest, ref_read)


def pack(a):
    if a.manifest.exists():
        prior = json.loads(a.manifest.read_bytes())
        assert prior["source_sha256"] == a.sha256 and prior["verified"]
        verify(prior)
        return
    assert a.part_limit >= 0 and a.sha256 and len(a.sha256) == 64
    assert a.source.is_file() and a.size > 0
    for folder in (a.primary, a.secondary, a.manifest.parent):
        folder.mkdir(exist_ok=True, parents=True)
    codec = Codec()
    digest = hashlib.sha256()
    parts = []
    before = a.source.stat()
    input_ctx = zstd_input(a.source) if a.zstd_source else nullcontext(None)
    ref = None
    if a.reference:
        assert a.reference.is_file() and len(a.reference_sha256) == 64 and a.reference_skip >= 0
        ref = dict(path=str(a.reference), raw_sha256=a.reference_sha256, skip=a.reference_skip)
    with input_ctx as process, reference_input(ref) as ref_read:
        with (nullcontext(process.stdout) if process else a.source.open("rb")) as src:
            left = a.size
            while left:
                i = len(parts)
                folder = a.primary if i < a.part_limit else a.secondary
                path = folder / f"{a.manifest.stem}.{i:03d}.zst"
                count = min(PART, left)
                compressed_hash = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=folder, prefix=path.name+".writing.", delete=False) as out:
                    tmp = Path(out.name)
                    try:
                        for pos in range(0, count, BLOCK):
                            size = min(BLOCK, count - pos)
                            raw = src.read(size)
                            assert len(raw) == size
                            digest.update(raw)
                            frame = codec.pack(block_bytes(raw, ref_read))
                            compressed_hash.update(frame)
                            out.write(frame)
                        out.flush()
                        os.fsync(out.fileno())
                        if path.exists():
                            assert file_hash(path) == compressed_hash.hexdigest()
                        else:
                            os.link(tmp, path)
                    finally:
                        tmp.unlink(missing_ok=True)
                parts.append(dict(path=str(path), source_bytes=count, packed_bytes=path.stat().st_size,
                                  packed_sha256=compressed_hash.hexdigest()))
                left -= count
            assert not src.read(1)
        if process:
            assert process.wait() == 0
    after = a.source.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_size) == (after.st_ino, after.st_mtime_ns, after.st_size)
    assert digest.hexdigest() == a.sha256
    index = dict(format="four_byte_planes_zstd_v1", block_bytes=BLOCK, source=str(a.source), reference=ref,
                 source_was_zstd=a.zstd_source, source_bytes=a.size, source_sha256=a.sha256,
                 archived_bytes=sum(p["packed_bytes"] for p in parts), parts=parts)
    verify(index)
    index["verified"] = True
    temp = a.manifest.with_suffix(".json.writing")
    with temp.open("w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(a.manifest)
    print(json.dumps({k:v for k,v in index.items() if k!="parts"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    encode = subs.add_parser("pack")
    encode.add_argument("--source", required=True, type=Path)
    encode.add_argument("--size", required=True, type=int)
    encode.add_argument("--sha256", required=True)
    encode.add_argument("--zstd-source", action="store_true")
    encode.add_argument("--reference", type=Path, help="Optional immutable zstd-compressed checkpoint for modular deltas.")
    encode.add_argument("--reference-sha256", help="SHA256 of the uncompressed reference bytes.")
    encode.add_argument("--reference-skip", type=int, default=0)
    encode.add_argument("--primary", required=True, type=Path)
    encode.add_argument("--secondary", required=True, type=Path)
    encode.add_argument("--part-limit", required=True, type=int)
    restore = subs.add_parser("restore")
    restore.add_argument("--dest", type=Path, required=True)
    inspect = subs.add_parser("verify")
    for p in (encode, restore, inspect):
        p.add_argument("--manifest", type=Path, required=True)
    a = parser.parse_args()
    if a.action == "pack":
        pack(a)
        return
    info = json.loads(a.manifest.read_bytes())
    assert info["verified"] and info["format"] == "four_byte_planes_zstd_v1"
    if a.action == "verify":
        verify(info)
    else:
        if os.path.lexists(a.dest):
            raise FileExistsError(a.dest)
        with tempfile.NamedTemporaryFile(dir=a.dest.parent, delete=False) as out:
            tmp = Path(out.name)
            try:
                verify(info, out)
                out.flush()
                os.fsync(out.fileno())
                os.link(tmp, a.dest)
            finally:
                tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
