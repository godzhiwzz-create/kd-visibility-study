#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

WORK_ROOT = Path('/root/kd_visibility')
WAIT_STATE = WORK_ROOT / 'mechanism' / 'faster_rcnn_kd_true' / 'phase4_m2_state.json'
STATE_PATH = WORK_ROOT / 'mechanism' / 'm13_dense_check' / 'm13_dense_state.json'
RUNNER_PID = WORK_ROOT / 'mechanism' / 'm13_dense_check' / 'm13_dense_runner.pid'
RUNNER = '/root/kd_visibility/mechanism/m13_dense_check/run_m13_dense_queue.py'
PYTHON = '/root/miniconda3/bin/python3'
GUARD_LOG = WORK_ROOT / 'logs' / 'm13_dense_guard.out'


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
        stat_path = Path('/proc') / str(pid) / 'stat'
        if stat_path.exists():
            parts = stat_path.read_text().split()
            if len(parts) >= 3 and parts[2] == 'Z':
                return False
        return True
    except OSError:
        return False


def runner_alive() -> bool:
    if not RUNNER_PID.exists():
        return False
    try:
        pid = int(RUNNER_PID.read_text().strip())
    except Exception:
        return False
    return pid_alive(pid)


def start_runner():
    proc = subprocess.Popen([PYTHON, RUNNER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    RUNNER_PID.write_text(str(proc.pid))
    log(f'launched runner pid={proc.pid}')


def wait_ready() -> bool:
    state = read_json(WAIT_STATE)
    if not state:
        return False
    return state.get('status') == 'completed'


def main():
    log('guard started')
    while True:
        wait_state = read_json(WAIT_STATE)
        if wait_state and wait_state.get('status') == 'failed':
            log('phase4_m2 failed, guard exiting instead of waiting forever')
            return
        if not wait_ready():
            log('waiting for phase4_m2 to complete')
            time.sleep(300)
            continue
        state = read_json(STATE_PATH) or {'status': 'pending'}
        if state.get('status') == 'completed':
            log('all tasks completed, guard exiting')
            return
        if state.get('status') == 'failed':
            log('state marked failed, guard exiting to avoid restart loop')
            return
        if not runner_alive():
            start_runner()
        time.sleep(60)

if __name__ == '__main__':
    main()
