#!/bin/bash
# 批量运行5×3实验矩阵 - v2版本

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/kd_visibility_claude:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export YOLO_VERBOSE=False
export ULTRALYTICS_AUTOINSTALL=False

PYTHON=/root/miniconda3/bin/python
PROJECT_ROOT=/root/autodl-tmp/kd_visibility_claude
OUTPUT_ROOT=$PROJECT_ROOT/outputs_v2

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
echo "5×3实验矩阵 - v2"
echo "总实验数: $TOTAL"
echo "每个实验轮数: $EPOCHS"
echo "输出目录: $OUTPUT_ROOT"
echo "开始时间: $(date)"
echo "=========================================="

# 遍历所有实验
for branch in "${BRANCHES[@]}"; do
    for vis in "${VISIBILITIES[@]}"; do
        CURRENT=$((CURRENT + 1))
        LOG_FILE="$OUTPUT_ROOT/logs/${branch}_${vis}.log"

        echo ""
        echo "[$CURRENT/$TOTAL] $branch / $vis"

        # 检查是否已完成
        if [ -f "$OUTPUT_ROOT/$branch/$vis/results.json" ]; then
            echo "  -> 已完成，跳过"
            continue
        fi

        # 运行实验
        echo "  -> 开始训练..."
        $PYTHON scripts/train_yolo_v2.py \
            --kd-branch $branch \
            --visibility $vis \
            --epochs $EPOCHS \
            --output-root $OUTPUT_ROOT \
            > $LOG_FILE 2>&1

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  -> FAILED (exit: $EXIT_CODE)"
            echo "$(date): $branch/$vis FAILED" >> $OUTPUT_ROOT/logs/failed.txt
        else
            MAP50=$(grep -o '"map50": [0-9.]*' $OUTPUT_ROOT/$branch/$vis/results.json 2>/dev/null | cut -d' ' -f2)
            echo "  -> 完成, mAP@50: $MAP50"
        fi

        sleep 2
    done
done

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "=========================================="

# 结果摘要
echo ""
echo "结果摘要:"
echo "------------------------------------------"
printf "%-20s %-12s %-12s %-12s\n" "KD Branch" "Light" "Moderate" "Heavy"
echo "------------------------------------------"

for branch in "${BRANCHES[@]}"; do
    printf "%-20s" "$branch"
    for vis in "${VISIBILITIES[@]}"; do
        RESULT_FILE="$OUTPUT_ROOT/$branch/$vis/results.json"
        if [ -f "$RESULT_FILE" ]; then
            MAP50=$(grep -o '"map50": [0-9.]*' $RESULT_FILE | head -1 | cut -d' ' -f2)
            printf " %-12s" "${MAP50:-ERROR}"
        else
            printf " %-12s" "FAILED"
        fi
    done
    echo ""
done
echo "------------------------------------------"
