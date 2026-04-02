#!/bin/bash
# KD训练守护脚本 - 自动重启

OUTPUT_ROOT=/root/autodl-tmp/kd_visibility_claude/outputs_v4
PYTHON=/root/miniconda3/bin/python

# 检查是否有训练进程
TRAIN_COUNT=$(ps aux | grep 'train_yolo_v4' | grep -v grep | wc -l)

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "$(date): 无训练进程，检查是否需要继续..." >> $OUTPUT_ROOT/guardian.log

    # 检查已完成数量
    COMPLETED=$(find $OUTPUT_ROOT -name 'results.json' 2>/dev/null | wc -l)

    if [ "$COMPLETED" -lt 15 ]; then
        echo "$(date): 已完成$COMPLETED/15，重启训练矩阵..." >> $OUTPUT_ROOT/guardian.log
        cd /root/autodl-tmp/kd_visibility_claude
        nohup bash run_matrix_v4.sh > $OUTPUT_ROOT/matrix.log 2>&1 &
    else
        echo "$(date): 全部完成!" >> $OUTPUT_ROOT/guardian.log
    fi
fi
