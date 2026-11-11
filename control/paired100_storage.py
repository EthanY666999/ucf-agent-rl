"""Manage explicitly listed, losslessly archived files for the paired-100 queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path("/data/runs/ucf_v3_budget10240_20261110")
STORE = ROOT / "paired100/storage"
SYS = Path("/home/vipuser/ucf-v3-weight-archives-20261112")
DATA = STORE / "archives"
PY = "/data/venvs/qwen35/bin/python"
GIB = 2**30
MODELS = {
    "S03_grpo_seed42": DATA,
    "S04_tree_random_seed42": SYS, "S05_ucf_seed42": SYS,
    "S06_gigpo_seed42": SYS, "S07_starpo_s_seed42": SYS,
}
BASE = Path("/home/vipuser/q35-final-models/S02_SFT_seed42/model.safetensors")


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(obj, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def free(path):
    return shutil.disk_usage(path).free


def free_gib():
    return {str(p): free(p)/GIB for p in (SYS, DATA)}


def model_index():
    models = {}
    for name, folder in MODELS.items():
        manifest = Path(str(folder / (name + ".f32delta")) + ".json")
        meta = read(manifest)
        parent = Path("/data/runs/retained_models" if name.startswith("S07_") else "/home/vipuser/q35-final-models")
        assert meta["source"] == str(parent / name / "model.safetensors")
        assert meta["base"] == str(BASE) and meta["archive"] == str(folder / (name + ".f32delta"))
        assert meta["verified"] and meta["decompressed_sha256"] == meta["source_sha256"]
        models[name] = meta
    return models


def release_models(commit):
    index_path = STORE / "models_plan.json"
    models = model_index()
    published = read(STORE / "release_authority.json")
    assert len(commit)==40 and all(c in "0123456789abcdef" for c in commit)
    assert published["commit"] == commit and published["plan_sha256"] == sha(index_path)
    assert read(index_path)["models"] == models
    assert {meta["base_sha256"] for meta in models.values()} == {sha(BASE)}
    before = free_gib()
    for meta in models.values():
        source, arc = Path(meta["source"]), Path(meta["archive"])
        pointer = source.parent / "ARCHIVED_WEIGHTS.json"
        if not source.exists():
            assert read(pointer)["source_sha256"] == meta["source_sha256"]
            continue
        assert source.is_file() and not source.is_symlink()
        assert sha(source) == meta["source_sha256"] and sha(arc) == meta["archive_sha256"]
        assert source.stat().st_size == meta["source_bytes"]
        used = subprocess.run(["lsof", "-t", "--", str(source)], capture_output=True)
        if used.stdout.strip() or used.returncode != 1:
            raise RuntimeError(f"In-use or uninspectable model: {source}")
        restore = [PY, str(STORE / "lossless_weight_archive.py"), "restore",
                   "--archive", str(arc), "--base", str(BASE), "--dest", str(source)]
        write(pointer, dict(meta, restore=restore, published_commit=commit))
        alias = SYS.parent / "q35-final-models" / source.parent.name / source.name
        if alias.is_symlink() and alias.resolve() == source:
            write(alias.parent / pointer.name, read(pointer))
        source.unlink()
    write(STORE/"models_released.json",dict(before_gib=before,after_gib=free_gib(),
                                          published_commit=commit,unix=time.time()))


def pack_anchor(method, primary_count):
    a = read(STORE.parent / "anchors50.json")["arms"][method]
    base = read(ROOT / "paired50_handoff/anchor.json")
    diff = base["source_bytes"] - a["size_bytes"]
    assert diff in (0,128)
    act = Path(a["path"])
    args = [PY, str(STORE / "pack_checkpoint.py"), "pack", "--manifest", str(STORE/f"{method}50.json"),
            "--source", str(act), "--size", str(a["size_bytes"]), "--sha256", a["sha256"],
            "--reference", base["staged_path"], "--reference-sha256", base["source_sha256"],
            "--reference-skip", str(diff),
            "--primary", str(DATA), "--secondary", str(SYS), "--part-limit", str(primary_count)]
    subprocess.run(args, check=True)


def release_anchor(method):
    history = read(STORE.parent / "anchors50.json")["arms"][method]
    raw = Path(history["path"])
    rolling = Path(history["rolling_path"])
    meta = read(STORE / f"{method}50.json")
    assert meta["verified"] and meta["source_sha256"] == history["sha256"] and meta["source_bytes"] == history["size_bytes"]
    assert meta["source"] == str(raw) and raw.name == "step0050.pt" and raw.parent.name == "retained_checkpoints"
    pointer = raw.with_suffix(".pt.archive.json")
    if not raw.exists():
        assert read(pointer)["source_sha256"] == history["sha256"]
        return
    assert not raw.is_symlink() and not os.path.samefile(raw, rolling) and raw.stat().st_nlink == 1
    assert read(rolling.parent/"status.json")["step"] >= 60 and rolling.stat().st_size > 0
    # The raw file hash and a full streaming reconstruction were matched before
    # publication of the part manifest. Recheck the immutable parts at release.
    for part in meta["parts"]:
        assert sha(Path(part["path"])) == part["packed_sha256"]
    assert sha(raw) == history["sha256"]
    restore = [PY, str(STORE/"pack_checkpoint.py"), "restore", "--manifest", str(STORE/f"{method}50.json"), "--dest", str(raw)]
    write(pointer,dict(source=str(raw),source_sha256=history["sha256"],source_bytes=history["size_bytes"],
                       manifest=str(STORE/f"{method}50.json"),restore=restore))
    before = free_gib()
    raw.unlink()
    write(STORE/f"{method}50_released.json",dict(before_gib=before,after_gib=free_gib(),unix=time.time()))


def objects():
    """Only immutable archive bytes are eligible, never active training files."""
    anchor = read(ROOT / "paired50_handoff/anchor.json")
    # The staged path is now the canonical real file or forwarding symlink.
    items = [(Path(anchor["staged_path"]), anchor["archive_sha256"], anchor["archive_bytes"])]
    for meta in model_index().values():
        items.append((Path(meta["archive"]), meta["archive_sha256"], meta["archive_bytes"]))
    for manifest in ("ucf50.json", "grpo50.json"):
        path = STORE/manifest
        if path.exists():
            items += [(Path(r["path"]), r["packed_sha256"], r["packed_bytes"]) for r in read(path)["parts"]]
    return items


def move_archive(path, expected, size, receiver):
    current = path.resolve(strict=True)
    assert current.stat().st_size == size and current.stat().st_nlink == 1
    if path.parent.stat().st_dev == receiver.stat().st_dev:
        dest = path
        assert path.is_symlink()
    else:
        dest = receiver / ("parked-" + expected[:12] + "-" + path.name)
    assert current != dest and (not dest.exists() or dest == path or sha(dest)==expected)
    if dest == path or not dest.exists():
        with tempfile.NamedTemporaryFile(dir=receiver, prefix=".moving-", delete=False) as out:
            tmp = Path(out.name)
            try:
                with current.open("rb") as inp:
                    shutil.copyfileobj(inp, out, 8*1024**2)
                out.flush()
                os.fsync(out.fileno())
                assert sha(tmp) == expected
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
    if path != dest:
        link = path.with_name(path.name+".forwarding")
        if os.path.lexists(link):
            assert link.is_symlink() and link.resolve()==dest
        else:
            link.symlink_to(dest)
        link.replace(path)
    assert path.resolve()==dest and sha(dest)==expected
    if current != path and current != dest:
        current.unlink()
    with (STORE/"moves.jsonl").open("a") as log:
        log.write(json.dumps(dict(canonical=str(path),storage=str(dest),sha256=expected,
                                  bytes=size,free_gib=free_gib(),unix=time.time()))+"\n")


def balance(active, target, idle_floor):
    a,b = (DATA,SYS) if active=="grpo" else (SYS,DATA)
    while free(a)<target*GIB:
        need = target*GIB-free(a)
        possible = [(path,h,size) for path,h,size in objects()
                    if path.stat().st_dev==a.stat().st_dev and size<free(b)-idle_floor*GIB]
        if not possible:
            raise RuntimeError(f"No immutable archive fits for {active} reserve {target}; free {free_gib()}")
        # Smallest one-file solution; otherwise move the largest available.
        chosen = min(possible, key=lambda v:(v[2]<need, abs(v[2]-need)))
        move_archive(*chosen, receiver=b)
    print(json.dumps(free_gib()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="op", required=True)
    subs.add_parser("inventory")
    release = subs.add_parser("release-models")
    release.add_argument("--published-commit", required=True)
    pack = subs.add_parser("pack-anchor")
    pack.add_argument("method", choices=["ucf","grpo"])
    pack.add_argument("--primary-count",type=int,required=True)
    pin = subs.add_parser("release-anchor")
    pin.add_argument("method", choices=["ucf","grpo"])
    move = subs.add_parser("reserve")
    move.add_argument("method",choices=["ucf","grpo"])
    move.add_argument("--target",type=float,required=True)
    move.add_argument("--idle-floor",type=float,default=6)
    a = parser.parse_args()
    with (STORE/"storage.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.op == "inventory":
            write(STORE/"models_plan.json", dict(models=model_index(), reserved_unarchived_base=str(BASE),
                 policy="Verified independent reconstruction of each original safetensors before replacing an inactive plain copy."))
        elif a.op=="release-models":
            release_models(a.published_commit)
        elif a.op=="pack-anchor":
            pack_anchor(a.method,a.primary_count)
        elif a.op=="release-anchor":
            release_anchor(a.method)
        else:
            balance(a.method,a.target,a.idle_floor)


if __name__ == "__main__":
    main()
