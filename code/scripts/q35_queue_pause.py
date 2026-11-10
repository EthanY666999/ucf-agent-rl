"""Honor an explicit user pause without signalling the current training process."""
import os
import time
from pathlib import Path
from q35_run_receipt import read, atomic, sha


def honor_user_pause(source, state=Path('/data/runs/queue_q35_hard'), runs=Path('/data/runs')):
    marker = state / 'pause_after_S05.json'
    if not marker.exists():
        return None
    request = read(marker)
    assert request.get('active') is True and request.get('pause_after') == 'S05', 'Invalid pause marker: inspect before any launch'
    run = runs / 'hard_queue/S05_ucf_seed42'
    verification = run / 'verification.json'
    checked = read(verification) if verification.exists() else {}
    complete = checked.get('passed') is True and checked.get('steps') == 400
    result = dict(status='paused_by_user_after_S05' if complete else 'pause_pending_S05_completion',
                  pause_after='S05', source=source, pid=os.getpid(), unix=time.time(),
                  S05_verified=complete, next_experiment='F03', launch_authorized=False,
                  pause_request=str(marker), pause_request_sha256=sha(marker),
                  resume_requires='new explicit user authorization')
    atomic(state / 'queue_pause_status.json', result)
    atomic(state / 'status.json', result)
    return result
