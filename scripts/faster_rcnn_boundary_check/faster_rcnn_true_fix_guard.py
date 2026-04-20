#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

WORK_ROOT = Path('/root/kd_visibility')
STATE_PATH = WORK_ROOT / 'scripts/faster_rcnn_boundary_check' / 'phase4_true_fix_state.json'
RUNNER_PID = WORK_ROOT / 'scripts/faster_rcnn_boundary_check' / 'phase4_true_fix_runner.pid'
RUNNER = '/root/kd_visibility/scripts/faster_rcnn_boundary_check/run_faster_rcnn_true_fix_queue.py'
PYTHON = '/root/miniconda3/bin/python3'
GUARD_LOG = WORK_ROOT / 'logs' / 'faster_rcnn_true_fix_guard.out'


def log(msg: str):
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GUARD_LOG.open('a') as f:
        f.write(f'[{time.strftime("%F %T")}] {msg}\n')


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runner_alive() -> bool:
    if RUNNER_PID.exists():
        try:
            pid = int(RUNNER_PID.read_text().strip())
            return pid_alive(pid)
        except Exception:
            return False
    return False


def start_runner():
    proc = subprocess.Popen([PYTHON, RUNNER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    RUNNER_PID.write_text(str(proc.pid))
    log(f'launched runner pid={proc.pid}')


def main():
    log('guard started')
    while True:
        state = read_json(STATE_PATH) or {'status': 'pending'}
        if state.get('status') == 'completed':
            log('all tasks completed, guard exiting')
            return
        if not runner_alive():
            start_runner()
        time.sleep(60)


if __name__ == '__main__':
    main()
