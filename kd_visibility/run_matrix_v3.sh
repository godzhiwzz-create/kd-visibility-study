#!/bin/bash
# 批量运行5×3实验矩阵 - v3修复版

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/kd_visibility_claude:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export YOLO_VERBOSE=False
export ULTRALYTICS_AUTOINSTALL=False

PYTHON=/root/miniconda3/bin/python
PROJECT_ROOT=/root/autodl-tmp/kd_visibility_claude
OUTPUT_ROOT=$PROJECT_ROOT/outputs_v3

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
echo "5×3实验矩阵 - v3 (修复版)"
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
        $PYTHON scripts/train_yolo_v3.py \
            --kd-branch $branch \
            --visibility $vis \
            --epochs $EPOCHS \
            --output-root $OUTPUT_ROOT \
            > $LOG_FILE 2>&1

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  -> FAILED (exit: $EXIT_CODE)"
            echo "$(date): $branch/$vis FAILED" >> $OUTPUT_ROOT/logs/failed.txt
            # 失败后等待一段时间再继续
            sleep 10
        else
            MAP50=$(grep -o '"map50": [0-9.]*' $OUTPUT_ROOT/$branch/$vis/results.json 2>/dev/null | head -1 | cut -d' ' -f2)
            echo "  -> 完成, mAP@50: $MAP50"
            sleep 2
        fi
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

# 保存完整结果
SUMMARY_FILE="$OUTPUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).json"
echo "{" > $SUMMARY_FILE
echo "  \"timestamp\": \"$(date -Iseconds)\"," >> $SUMMARY_FILE
echo "  \"total_experiments\": $TOTAL," >> $SUMMARY_FILE
echo "  \"completed\": $(ls $OUTPUT_ROOT/*/results.json 2>/dev/null | wc -l)," >> $SUMMARY_FILE
echo "  \"results\": {" >> $SUMMARY_FILE

FIRST_BRANCH=1
for branch in "${BRANCHES[@]}"; do
    if [ $FIRST_BRANCH -eq 0 ]; then
        echo "," >> $SUMMARY_FILE
    fi
    FIRST_BRANCH=0
    echo -n "    \"$branch\": {" >> $SUMMARY_FILE

    FIRST_VIS=1
    for vis in "${VISIBILITIES[@]}"; do
        if [ $FIRST_VIS -eq 0 ]; then
            echo -n ", " >> $SUMMARY_FILE
        fi
        FIRST_VIS=0

        RESULT_FILE="$OUTPUT_ROOT/$branch/$vis/results.json"
        if [ -f "$RESULT_FILE" ]; then
            MAP50=$(grep -o '"map50": [0-9.]*' $RESULT_FILE | head -1 | cut -d' ' -f2)
            echo -n "\"$vis\": $MAP50" >> $SUMMARY_FILE
        else
            echo -n "\"$vis\": null" >> $SUMMARY_FILE
        fi
    done
    echo -n "}" >> $SUMMARY_FILE
done
echo "" >> $SUMMARY_FILE
echo "  }" >> $SUMMARY_FILE
echo "}" >> $SUMMARY_FILE

echo ""
echo "摘要已保存: $SUMMARY_FILE"
