#!/bin/bash
# 批量运行5×3实验矩阵

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/kd_visibility_claude:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export YOLO_VERBOSE=False

PYTHON=/root/miniconda3/bin/python
PROJECT_ROOT=/root/autodl-tmp/kd_visibility_claude
OUTPUT_ROOT=$PROJECT_ROOT/outputs

cd $PROJECT_ROOT

# 定义实验矩阵
BRANCHES=("student_only" "logit_only" "feature_only" "attention_only" "localization_only")
VISIBILITIES=("light" "moderate" "heavy")
EPOCHS=100

TOTAL=$(( ${#BRANCHES[@]} * ${#VISIBILITIES[@]} ))
CURRENT=0

# 创建日志目录
mkdir -p $OUTPUT_ROOT/logs

echo "=========================================="
echo "开始运行5×3实验矩阵"
echo "总实验数: $TOTAL"
echo "每个实验轮数: $EPOCHS"
echo "开始时间: $(date)"
echo "=========================================="

# 遍历所有实验
for branch in "${BRANCHES[@]}"; do
    for vis in "${VISIBILITIES[@]}"; do
        CURRENT=$((CURRENT + 1))
        LOG_FILE="$OUTPUT_ROOT/logs/${branch}_${vis}.log"

        echo ""
        echo "[$CURRENT/$TOTAL] 运行: $branch / $vis"
        echo "日志: $LOG_FILE"
        echo "开始: $(date)"

        # 检查是否已完成
        if [ -f "$OUTPUT_ROOT/kd_visibility_yolo/$branch/$vis/results.json" ]; then
            echo "  -> 已存在结果，跳过"
            continue
        fi

        # 运行实验
        $PYTHON scripts/train_yolo_kd.py \
            --kd-branch $branch \
            --visibility $vis \
            --epochs $EPOCHS \
            --output-root $OUTPUT_ROOT \
            > $LOG_FILE 2>&1

        EXIT_CODE=$?
        echo "结束: $(date), 退出码: $EXIT_CODE"

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  -> 实验失败!"
            echo "$(date): $branch/$vis FAILED" >> $OUTPUT_ROOT/logs/failed.txt
        else
            echo "  -> 实验完成"
        fi

        # 短暂休息让GPU降温
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "=========================================="

# 生成结果摘要
echo ""
echo "实验结果摘要:"
echo "------------------------------------------"
printf "%-20s %-12s %-12s %-12s\n" "KD Branch" "Light" "Moderate" "Heavy"
echo "------------------------------------------"

for branch in "${BRANCHES[@]}"; do
    printf "%-20s" "$branch"
    for vis in "${VISIBILITIES[@]}"; do
        RESULT_FILE="$OUTPUT_ROOT/kd_visibility_yolo/$branch/$vis/results.json"
        if [ -f "$RESULT_FILE" ]; then
            MAP50=$($PYTHON -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['metrics']['map50']:.4f}\")" 2>/dev/null || echo "ERROR")
            printf " %-12s" "$MAP50"
        else
            printf " %-12s" "MISSING"
        fi
    done
    echo ""
done
echo "------------------------------------------"
