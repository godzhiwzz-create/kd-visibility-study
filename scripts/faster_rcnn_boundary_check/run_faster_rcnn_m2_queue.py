#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

WORK_ROOT = Path('/root/kd_visibility')
TASKS_PATH = WORK_ROOT / 'scripts/faster_rcnn_boundary_check' / 'phase4_m2_tasks.json'
STATE_PATH = WORK_ROOT / 'scripts/faster_rcnn_boundary_check' / 'phase4_m2_state.json'
LOG_DIR = WORK_ROOT / 'logs' / 'faster_rcnn_phase4_m2'
RUNNER_PID = WORK_ROOT / 'scripts/faster_rcnn_boundary_check' / 'phase4_m2_runner.pid'
PYTHON = '/root/miniconda3/bin/python3'
SCRIPT = '/root/kd_visibility/scripts/faster_rcnn_boundary_check/train_faster_rcnn_true_kd.py'
SUMMARY_SCRIPT = '/root/kd_visibility/scripts/faster_rcnn_boundary_check/summarize_phase4_m2.py'
OUTPUT_DIR = 'outputs_faster_rcnn_true_phase4_m2'
MAX_ATTEMPTS = 3
DEFAULT_TEACHER = '/root/kd_visibility/outputs_faster_rcnn_true_phase4_fix/occ_0.0_student_only_deg_0.0_seed_42/best_model_student.pth'


def load_tasks():
    return json.loads(TASKS_PATH.read_text())


def default_state():
    return {
        'status': 'pending',
        'artifacts': {
            'teacher_path': DEFAULT_TEACHER,
        },
        'tasks': load_tasks(),
        'updated_at': time.time(),
    }


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    state = default_state()
    save_state(state)
    return state


def save_state(state):
    state['updated_at'] = time.time()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')


def task_output_root(task):
    exp_name = f"occ_{task['occlusion_ratio']}_{task['kd_branch']}_deg_{float(task['beta'])}_seed_{task['seed']}"
    return WORK_ROOT / OUTPUT_DIR / exp_name


def task_result_path(task):
    return task_output_root(task) / 'results.json'


def mark_completed_from_disk(state):
    for task in state['tasks']:
        result_path = task_result_path(task)
        if result_path.exists():
            task['status'] = 'completed'
            task['last_error'] = None


def build_command(task, state):
    cmd = [
        PYTHON,
        SCRIPT,
        '--occlusion_ratio',
        str(task['occlusion_ratio']),
        '--beta',
        str(task['beta']),
        '--kd_branch',
        task['kd_branch'],
        '--epochs',
        str(task['epochs']),
        '--batch_size',
        str(task['batch_size']),
        '--workers',
        str(task['workers']),
        '--prep_workers',
        str(task['prep_workers']),
        '--seed',
        str(task['seed']),
        '--output_dir',
        OUTPUT_DIR,
    ]
    if 'kd_weight' in task:
        cmd.extend(['--kd_weight', str(task['kd_weight'])])
    if 'temperature' in task:
        cmd.extend(['--temperature', str(task['temperature'])])
    teacher_key = task.get('teacher_from_artifact')
    if teacher_key:
        teacher_path = state.get('artifacts', {}).get(teacher_key)
        if not teacher_path:
            raise RuntimeError(f'missing teacher artifact: {teacher_key}')
        cmd.extend(['--teacher_path', teacher_path])
    return cmd


def summarize_results():
    return subprocess.run([PYTHON, SUMMARY_SCRIPT], check=False)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNNER_PID.write_text(str(os.getpid()))
    state = load_state()
    mark_completed_from_disk(state)
    save_state(state)
    try:
        while True:
            state = load_state()
            mark_completed_from_disk(state)
            pending = [t for t in state['tasks'] if t['status'] != 'completed']
            if not pending:
                summary_proc = summarize_results()
                if summary_proc.returncode == 0:
                    state['status'] = 'completed'
                else:
                    state['status'] = 'failed'
                    state['last_error'] = f'summary_failed returncode={summary_proc.returncode}'
                save_state(state)
                return

            task = pending[0]
            if int(task.get('attempts', 0)) >= MAX_ATTEMPTS:
                task['status'] = 'failed'
                state['status'] = 'failed'
                save_state(state)
                raise RuntimeError(f"task failed too many times: {task['name']}")

            task['status'] = 'running'
            task['attempts'] = int(task.get('attempts', 0)) + 1
            task['last_error'] = None
            log_path = LOG_DIR / f"{task['name']}.log"
            task['log_path'] = str(log_path)
            save_state(state)

            cmd = build_command(task, state)
            with log_path.open('a') as logf:
                logf.write(f"\n=== START {time.strftime('%F %T')} ===\n")
                logf.write('CMD: ' + ' '.join(cmd) + '\n')
                logf.flush()
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                ret = proc.wait()

            state = load_state()
            current = next(t for t in state['tasks'] if t['name'] == task['name'])
            if ret == 0 and task_result_path(current).exists():
                current['status'] = 'completed'
                current['last_error'] = None
                state['status'] = 'running'
            else:
                current['status'] = 'pending'
                tail = ''
                if log_path.exists():
                    tail = '\n'.join(log_path.read_text(errors='ignore').splitlines()[-60:])
                current['last_error'] = f'returncode={ret}\n{tail[-4000:]}'
                if 'out of memory' in tail.lower() and current['batch_size'] > 1:
                    current['batch_size'] = max(1, current['batch_size'] // 2)
                state['status'] = 'running'
            save_state(state)
    finally:
        if RUNNER_PID.exists() and RUNNER_PID.read_text().strip() == str(os.getpid()):
            RUNNER_PID.unlink()


if __name__ == '__main__':
    main()
