"""
批量运行5×3实验矩阵的脚本
支持顺序执行和并行执行
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import KDBranch, VisibilityLevel


# 实验矩阵定义
KD_BRANCHES = [
    'student_only',
    'logit_only',
    'feature_only',
    'attention_only',
    'localization_only'
]

VISIBILITY_LEVELS = ['light', 'moderate', 'heavy']


def run_single_experiment(
    kd_branch: str,
    visibility: str,
    epochs: int = 50,
    output_root: str = "outputs",
    gpu_id: int = 0
) -> Dict:
    """
    运行单个实验

    Returns:
        包含实验结果的字典
    """
    print(f"\n{'='*60}")
    print(f"启动实验: {kd_branch} / {visibility}")
    print(f"{'='*60}\n")

    # 设置GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 构建命令
    cmd = [
        sys.executable,
        '-m', 'kd_visibility.scripts.train',
        '--kd-branch', kd_branch,
        '--visibility', visibility,
        '--epochs', str(epochs),
        '--output-root', output_root
    ]

    # 运行训练
    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        # 解析输出获取最佳mIoU
        best_miou = 0.0
        for line in result.stdout.split('\n'):
            if '最佳验证mIoU' in line or 'best' in line.lower():
                try:
                    best_miou = float(line.split(':')[-1].strip().replace(')', ''))
                except:
                    pass

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            'kd_branch': kd_branch,
            'visibility': visibility,
            'status': 'success',
            'best_miou': best_miou,
            'duration': duration,
            'error': None
        }

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"实验失败: {kd_branch} / {visibility}")
        print(f"错误: {e.stderr}")

        return {
            'kd_branch': kd_branch,
            'visibility': visibility,
            'status': 'failed',
            'best_miou': 0.0,
            'duration': duration,
            'error': str(e)
        }


def run_experiment_matrix(
    epochs: int = 50,
    output_root: str = "outputs",
    parallel: bool = False,
    max_workers: int = 1,
    resume: bool = False
):
    """
    运行完整的5×3实验矩阵

    Args:
        epochs: 每实验的轮数
        output_root: 输出根目录
        parallel: 是否并行执行
        max_workers: 并行工作数
        resume: 是否跳过已完成的实验
    """
    # 创建实验列表
    experiments = []
    for branch in KD_BRANCHES:
        for vis in VISIBILITY_LEVELS:
            experiments.append((branch, vis))

    print(f"总共 {len(experiments)} 个实验需要运行")

    # 检查已完成的实验
    if resume:
        pending_experiments = []
        for branch, vis in experiments:
            exp_dir = Path(output_root) / "kd_visibility_study" / branch / vis
            if (exp_dir / 'training_history.json').exists():
                print(f"跳过已完成实验: {branch} / {vis}")
            else:
                pending_experiments.append((branch, vis))
        experiments = pending_experiments
        print(f"剩余 {len(experiments)} 个实验需要运行")

    # 运行实验
    results = []

    if parallel and max_workers > 1:
        # 并行执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (branch, vis) in enumerate(experiments):
                gpu_id = i % max_workers
                future = executor.submit(
                    run_single_experiment,
                    branch, vis, epochs, output_root, gpu_id
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"\n完成: {result['kd_branch']} / {result['visibility']} "
                      f"-> mIoU: {result['best_miou']:.4f}")
    else:
        # 顺序执行
        for branch, vis in experiments:
            result = run_single_experiment(branch, vis, epochs, output_root)
            results.append(result)

    # 保存结果摘要
    save_results_summary(results, output_root)

    # 打印摘要
    print_summary(results)


def save_results_summary(results: List[Dict], output_root: str):
    """保存结果摘要"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': results
    }

    output_path = Path(output_root) / 'experiment_summary.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果摘要已保存: {output_path}")


def print_summary(results: List[Dict]):
    """打印结果摘要"""
    print(f"\n{'='*80}")
    print("实验矩阵结果摘要")
    print(f"{'='*80}")

    # 构建结果表格
    print("\n{:<20} {:<12} {:<12} {:<12}".format("KD Branch", "Light", "Moderate", "Heavy"))
    print("-" * 60)

    for branch in KD_BRANCHES:
        row = [branch]
        for vis in VISIBILITY_LEVELS:
            result = next(
                (r for r in results
                 if r['kd_branch'] == branch and r['visibility'] == vis),
                None
            )
            if result and result['status'] == 'success':
                row.append(f"{result['best_miou']:.4f}")
            else:
                row.append("FAILED")
        print("{:<20} {:<12} {:<12} {:<12}".format(*row))

    # 统计
    successful = sum(1 for r in results if r['status'] == 'success')
    total = len(results)
    print(f"\n成功率: {successful}/{total} ({100*successful/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='批量运行KD实验矩阵')
    parser.add_argument('--epochs', type=int, default=50,
                        help='每实验的训练轮数')
    parser.add_argument('--output-root', type=str, default='outputs',
                        help='输出根目录')
    parser.add_argument('--parallel', action='store_true',
                        help='是否并行执行')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='并行工作数')
    parser.add_argument('--resume', action='store_true',
                        help='跳过已完成的实验')

    args = parser.parse_args()

    run_experiment_matrix(
        epochs=args.epochs,
        output_root=args.output_root,
        parallel=args.parallel,
        max_workers=args.max_workers,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
