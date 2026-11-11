"""Preserve audited step-50 checkpoints and maintain the active writers' reserve."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from paired100_storage import STORE, DATA, SYS, PY, GIB, free, free_gib, read, write


def status(phase, **details):
    write(STORE/"guard_status.json",dict(phase=phase,free_gib=free_gib(),unix=time.time(),**details))


def wait_for(check, phase):
    status(phase)
    while not check():
        time.sleep(30)


def op(*args):
    subprocess.run([PY,str(STORE/"paired100_storage.py"),*args],check=True)


def rolled(method, a):
    anchor, rolling = Path(a["path"]),Path(a["rolling_path"])
    if not anchor.exists():
        return (STORE / f"{method}50_released.json").exists()
    return (rolling.is_file() and not os.path.samefile(anchor,rolling)
            and read(rolling.parent/"status.json").get("step",0)>=60)


def distribute():
    # Both arms have the same architecture and optimizer. The existing UCF
    # part sizes predict disk placement only. The GRPO hash is checked in full.
    sizes=[p["packed_bytes"] for p in read(STORE/"ucf50.json")["parts"]]
    a,b = free(DATA),free(SYS)
    predictions=[min(a-1.15*sum(sizes[:n]),b-1.15*sum(sizes[n:])) for n in range(len(sizes)+1)]
    n=max(range(len(predictions)),key=predictions.__getitem__)
    if predictions[n]<4*GIB:
        raise RuntimeError(f"Insufficient temporary encoding room: {free_gib()}")
    return n


def main():
    owner=(STORE/"guard.lock").open("a")
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    anchors=read(STORE.parent/"anchors50.json")["arms"]
    wait_for(lambda:(STORE/"models_released.json").exists(),"waiting_for_verified_model_archives")
    if not (STORE/"ucf50_released.json").exists():
        if not (STORE/"ucf50.json").exists():
            status("packing_ucf50")
            op("pack-anchor","ucf","--primary-count","10")
        if not rolled("ucf",anchors["ucf"]):
            op("reserve","ucf","--target","32.5","--idle-floor","7")
        wait_for(lambda:rolled("ucf",anchors["ucf"]),"waiting_for_ucf60_checkpoint")
        op("release-anchor","ucf")
        op("reserve","ucf","--target","33","--idle-floor","7")
    if not (STORE/"grpo50_released.json").exists():
        wait_for(lambda:rolled("grpo",anchors["grpo"]),"waiting_for_grpo60_checkpoint")
        if not (STORE/"grpo50.json").exists():
            n=distribute()
            status("packing_grpo50",data_parts=n)
            op("pack-anchor","grpo","--primary-count",str(n))
        op("release-anchor","grpo")
    status("balancing_grpo")
    op("reserve","grpo","--target","32.25","--idle-floor","6")
    # A 21 GiB temporary checkpoint is expected while torch.save is active.
    # Check normal headroom between atomic writes, never react to that transient.
    run=Path(anchors["grpo"]["rolling_path"]).parent
    while not (run/"stage_0100.json").is_file():
        if not (run/"checkpoint.tmp").exists() and free(DATA)<31.25*GIB:
            snapshot=read(run/"status.json")
            step=snapshot.get("current_step",0)
            if step and step%10 not in (0,1):
                op("reserve","grpo","--target","32","--idle-floor","5.5")
        time.sleep(60)
    status("both_step50_anchors_preserved_and_grpo100_checkpointed")


if __name__=="__main__":
    try:
        main()
    except BlockingIOError:
        raise
    except Exception as error:
        status("needs_storage_review",error=repr(error))
        raise
