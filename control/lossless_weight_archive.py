"""Lossless storage for inactive same-layout FP32 safetensors; CPU only."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np


MAGIC = b"UCF-F32-DELTA-1\n"
CHUNK_BYTES = 8 * 1024 ** 2
RECORD = struct.Struct("<II32s")


def exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError("Truncated file")
    return data


def header(f):
    size_bytes = exact(f, 8)
    size, = struct.unpack("<Q", size_bytes)
    if size > 32 * 1024 ** 2:
        raise ValueError("Invalid safetensors header size")
    return size_bytes + exact(f, size)


def layout(h):
    result = json.loads(h[8:])
    result.pop("__metadata__", None)
    if any(entry["dtype"] != "F32" for entry in result.values()):
        raise ValueError("Only dense FP32 safetensors supported")
    return result


class Codec:
    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library("zstd"))
        for key, args in {
            "ZSTD_compressBound": [ctypes.c_size_t],
            "ZSTD_compress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int],
            "ZSTD_decompress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t],
            "ZSTD_isError": [ctypes.c_size_t],
        }.items():
            fn = getattr(self.lib, key)
            fn.argtypes = args
            fn.restype = ctypes.c_uint if key == "ZSTD_isError" else ctypes.c_size_t

    def pack(self, data):
        cap = self.lib.ZSTD_compressBound(len(data))
        out = ctypes.create_string_buffer(cap)
        n = self.lib.ZSTD_compress(out, cap, data, len(data), 5)
        if self.lib.ZSTD_isError(n):
            raise ValueError("zstd encode error")
        return out.raw[:n]

    def unpack(self, data, size):
        out = ctypes.create_string_buffer(size)
        n = self.lib.ZSTD_decompress(out, size, data, len(data))
        if n != size:
            raise ValueError("zstd decode size mismatch")
        return out.raw


def delta_bytes(src, base):
    a = np.frombuffer(src, dtype="<u4")
    b = np.frombuffer(base, dtype="<u4")
    delta = (a - b).view("<i4")
    zigzag = (delta.view("<u4") << 1) ^ (delta >> 31).view("<u4")
    return zigzag.view("u1").reshape(-1, 4).T.tobytes()


def original_bytes(packed, base):
    zigzag = np.frombuffer(packed, dtype="u1").reshape(4, -1).T.copy().reshape(-1).view("<u4")
    delta = (zigzag >> 1) ^ (np.uint32(0) - (zigzag & np.uint32(1)))
    return (delta + np.frombuffer(base, dtype="<u4")).tobytes()


def fingerprint(f):
    st = os.fstat(f.fileno())
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def sha(data):
    return hashlib.sha256(data)


def file_sha(path):
    h = sha(b"")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def write_all(file, data):
    file.write(data)
    file.flush()
    os.fsync(file.fileno())


def sync_dir(dest):
    fd = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(temp, dest):
    # link refuses to replace an existing output from any concurrent invocation.
    os.link(temp, dest)
    temp.unlink()
    sync_dir(dest.parent)


def encode(args):
    src, base, archive = args.source.resolve(strict=True), args.base.resolve(strict=True), args.archive
    receipt = Path(str(archive) + ".json")
    if archive.exists() or receipt.exists() or src == base:
        raise ValueError("New destination and distinct source/base required")
    codec = Codec()
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=archive.name + ".", delete=False) as output:
        temp = Path(output.name)
        try:
            with src.open("rb") as srcf, base.open("rb") as basef:
                fs, fb = fingerprint(srcf), fingerprint(basef)
                sh, bh = header(srcf), header(basef)
                if layout(sh) != layout(bh):
                    raise ValueError("Tensor layouts differ")
                hs, hb = sha(sh), sha(bh)
                output.write(MAGIC + sh)
                chunks = 0
                while raw := srcf.read(CHUNK_BYTES):
                    if len(raw) % 4:
                        raise ValueError("Invalid FP32 payload")
                    baseraw = exact(basef, len(raw))
                    encoded = codec.pack(delta_bytes(raw, baseraw))
                    output.write(RECORD.pack(len(raw), len(encoded), sha(raw).digest()))
                    output.write(encoded)
                    hs.update(raw)
                    hb.update(baseraw)
                    chunks += 1
                if basef.read(1) or fs != fingerprint(srcf) or fb != fingerprint(basef):
                    raise ValueError("Input changed during encoding")
                write_all(output, b"")
                manifest = dict(format_version=1, source=str(src), base=str(base), archive=str(archive.resolve()),
                                source_bytes=fs[2], source_sha256=hs.hexdigest(), base_sha256=hb.hexdigest(),
                                archive_bytes=temp.stat().st_size, archive_sha256=file_sha(temp), chunks=chunks)
            verify(temp, base, manifest)
            manifest.update(verified=True, verified_unix=time.time(), decompressed_sha256=manifest["source_sha256"])
            publish(temp, archive)
            with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as f:
                receipt_tmp = Path(f.name)
                write_all(f, (json.dumps(manifest, indent=2) + "\n").encode())
            publish(receipt_tmp, receipt)
            print(json.dumps(manifest), flush=True)
        finally:
            temp.unlink(missing_ok=True)


def verify(archive, base, manifest, dest_file=None):
    codec = Codec()
    with archive.open("rb") as af, base.open("rb") as bf:
        if exact(af, len(MAGIC)) != MAGIC:
            raise ValueError("Invalid archive format")
        sh, bh = header(af), header(bf)
        if layout(sh) != layout(bh):
            raise ValueError("Invalid base tensor layout")
        hs, hb, ha = sha(sh), sha(bh), sha(MAGIC + sh)
        size, chunks = len(sh), 0
        if dest_file:
            dest_file.write(sh)
        while packet := af.read(RECORD.size):
            nr, nz, expected = RECORD.unpack(packet)
            if nr == 0 or nr > CHUNK_BYTES or nr % 4 or nz == 0 or nz > codec.lib.ZSTD_compressBound(nr):
                raise ValueError("Invalid archive chunk")
            comp, src_base = exact(af, nz), exact(bf, nr)
            raw = original_bytes(codec.unpack(comp, nr), src_base)
            if sha(raw).digest() != expected:
                raise ValueError(f"Reconstruction failed on chunk {chunks}")
            hs.update(raw)
            hb.update(src_base)
            ha.update(packet)
            ha.update(comp)
            size += nr
            chunks += 1
            if dest_file:
                dest_file.write(raw)
        if bf.read(1) or (size, chunks, hs.hexdigest(), hb.hexdigest(), ha.hexdigest()) != (
                    manifest["source_bytes"], manifest["chunks"], manifest["source_sha256"],
                    manifest["base_sha256"], manifest["archive_sha256"]):
            raise ValueError("Full archive/source/base identity mismatch")
        if dest_file:
            write_all(dest_file, b"")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="command", required=True)
    for mode in ("encode", "verify", "restore"):
        sub = actions.add_parser(mode)
        sub.add_argument("--base", required=True, type=Path)
        sub.add_argument("--archive", required=True, type=Path)
        if mode == "encode":
            sub.add_argument("--source", required=True, type=Path)
        if mode == "restore":
            sub.add_argument("--dest", required=True, type=Path)
    args = p.parse_args()
    if args.command == "encode":
        encode(args)
        return
    manifest = json.loads(Path(str(args.archive) + ".json").read_bytes())
    if args.command == "verify":
        verify(args.archive, args.base, manifest)
        print(manifest["source_sha256"])
        return
    if os.path.lexists(args.dest):
        raise FileExistsError(args.dest)
    with tempfile.NamedTemporaryFile(dir=args.dest.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            verify(args.archive, args.base, manifest, out)
            publish(temp, args.dest)
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
