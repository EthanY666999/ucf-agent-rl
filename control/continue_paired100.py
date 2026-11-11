"""Continue each of the two archived Sokoban runs through total step 100."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/data/runs/ucf_v3_budget10240_20261110")
WORK = ROOT / "paired100"
SOURCE_HASH = "f3929f9ac2c73fad3432565b09efb566894401c965c9095e3f346a9611c3d6e0"
CPU = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    tmp = path.with_suffix(".writing")
    with tmp.open("w") as stream:
        json.dump(obj, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status(phase, **fields):
    save(WORK / "status.json", dict(phase=phase, unix=time.time(), **fields))


def plan():
    source = ROOT / "plan.json"
    assert digest(source) == SOURCE_HASH
    assert read(ROOT / "reports/paired_export_0050.json")["passed"]
    old = read(source)
    selected = [e for e in old["entries"] if e["stop_step"] == 100]
    assert [(e["method"], e["kind"]) for e in selected] == [("ucf", "train"), ("grpo", "train")]
    for e in selected:
        previous = next(x for x in old["entries"] if x["id"] == e["id"] and x["stop_step"] == 50)
        assert dict(previous, stop_step=100) == e
        assert digest(Path(e["config"])) == e["config_sha256"]
        cfg = read(Path(e["config"]))
        assert (cfg["max_response_tokens"], cfg["max_turns"], cfg["eval_every"], cfg["ucf_variant"]) == (10240, 10, 25, "uniform")
        assert read(Path(e["output"]) / "stage_0050.json")["passed"]
    new = dict(old, entries=selected, state=str(WORK), source_plan_sha256=SOURCE_HASH,
               revision="authorized_paired50_to100_only", stop_after_pair=True,
               intended_review="50 further updates of each own checkpoint. Fixed validation at 75 and 100; stop at 100.")
    for e in selected:
        method = e["method"]
        save(WORK / f"{method}_only.json", dict(new, state=str(WORK / method), entries=[e]))
    save(WORK / "bounded_plan.json", new)
    return new


def main():
    WORK.mkdir(exist_ok=True)
    own = (WORK / "owner.lock").open("a")
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    new = plan()
    if sys.argv[1:] == ["--prepare"]:
        return
    assert not sys.argv[1:]
    # Inherit the same queue/GPU lock pair directly into every signed training
    # manager. One owner spans the boundary and cannot consume old plan entries.
    locks = [Path(p).open("a") for p in new["locks"]]
    for handle in locks:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fds = tuple(f.fileno() for f in locks)
    env = dict(os.environ, UCF_INHERITED_QUEUE_LOCK_FD=str(fds[0]), UCF_INHERITED_GPU_LOCK_FD=str(fds[1]))
    env.pop("CUDA_VISIBLE_DEVICES", None)
    anchors = read(WORK / "anchors50.json")["arms"]
    for e in new["entries"]:
        a = anchors[e["method"]]
        assert a["step"] == 50 and len(a["sha256"]) == 64
        assert Path(a["path"]).stat().st_size == a["size_bytes"] > 0
        out = Path(e["output"])
        if not (WORK / e["method"] / "attempts").exists():
            assert os.path.samefile(out / "checkpoint.pt", a["path"])
            assert read(out / "status.json")["step"] == 50
    save(WORK / "launch.json", dict(pid=os.getpid(), locks=new["locks"], unix=time.time(), entries=new["entries"]))
    for e in new["entries"]:
        method = e["method"]
        out = Path(e["output"])
        boundary = out / "stage_0100.json"
        if not boundary.exists():
            status("running", method=method, stop_after_step=100, available_gib=shutil.disk_usage(out).free/2**30)
            job = WORK / f"{method}_only.json"
            with (WORK / f"{method}_manager.log").open("a") as log:
                subprocess.run([new["python"], "-u", new["manager_script"], str(job)],
                               env=env, pass_fds=fds, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not boundary.exists() or not read(boundary)["passed"]:
            raise RuntimeError(f"{method}: missing verified step 100; inspect {WORK / method / 'failures'}")
        status("archiving_fixed_validation", method=method, step=100)
        subprocess.run([new["python"], str(WORK/"archive_paired100.py"), "--arm", method], check=True, env=CPU)
    subprocess.run([new["python"], str(WORK/"archive_paired100.py")], check=True, env=CPU)
    status("both_100_verified_and_archived", stop_after_step=100)


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        raise
    except Exception as exc:
        if WORK.is_dir():
            status("needs_recovery", error=repr(exc))
        raise
