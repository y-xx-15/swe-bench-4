#!/bin/bash
set -e

# ========================= 环境配置 =========================
export OPENAI_API_KEY="sk-V09Av97hiMfatQyzRUhpjrvovwLStSHLrSn0NE5tQHYACjiN"
export OPENAI_API_BASE="https://api.key77qiqi.cn/v1"

PROJECT_DIR="/home/xxy/SWE-bench-4"
OUTPUT_ENHANCED="$PROJECT_DIR/outputs/enhanced_patch_pipeline_enhanced"
OUTPUT_BASELINE="$PROJECT_DIR/outputs/enhanced_patch_pipeline_baseline"
LOG_DIR="$PROJECT_DIR/experiment_logs"

MODEL="gpt-4o"

mkdir -p "$LOG_DIR" "$OUTPUT_ENHANCED" "$OUTPUT_BASELINE"

# ========================= 77 个 SymPy 实例 =========================
INSTANCE_IDS=(
    sympy__sympy-14317
    sympy__sympy-12454
    sympy__sympy-12481
    sympy__sympy-14396
    sympy__sympy-14774
    sympy__sympy-14817
    sympy__sympy-15011
    sympy__sympy-15308
    sympy__sympy-15345
    sympy__sympy-15346
    sympy__sympy-15609
    sympy__sympy-15678
    sympy__sympy-16106
    sympy__sympy-16281
    sympy__sympy-16503
    sympy__sympy-16792
    sympy__sympy-16988
    sympy__sympy-17022
    sympy__sympy-17139
    sympy__sympy-17630
    sympy__sympy-17655
    sympy__sympy-18057
    sympy__sympy-18087
    sympy__sympy-18189
    sympy__sympy-18199
    sympy__sympy-18532
    sympy__sympy-18621
    sympy__sympy-18698
    sympy__sympy-18835
    sympy__sympy-19007
    sympy__sympy-19254
    sympy__sympy-19487
    sympy__sympy-20049
    sympy__sympy-20154
    sympy__sympy-20212
    sympy__sympy-20322
    sympy__sympy-20442
    sympy__sympy-20639
    sympy__sympy-21055
    sympy__sympy-21171
    sympy__sympy-21379
    sympy__sympy-21612
    sympy__sympy-21614
    sympy__sympy-21627
    sympy__sympy-21847
    sympy__sympy-22005
    sympy__sympy-22714
    sympy__sympy-22840
    sympy__sympy-23117
    sympy__sympy-23191
    sympy__sympy-23262
    sympy__sympy-24066
    sympy__sympy-24102
    sympy__sympy-24152
    sympy__sympy-24213
    sympy__sympy-24909
)


# ========================= 主循环 =========================
TOTAL=${#INSTANCE_IDS[@]}
CURRENT=0
START_TIME=$(date +%s)

for INSTANCE_ID in "${INSTANCE_IDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    INSTANCE_START=$(date +%s)
    LOG_FILE="$LOG_DIR/${INSTANCE_ID}.log"
    
    echo ""
    echo "=============================================="
    echo "████ [进度: $CURRENT / $TOTAL] 开始处理: $INSTANCE_ID"
    echo "████ 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
    
    echo "开始处理 $INSTANCE_ID" >> "$LOG_FILE"
    date >> "$LOG_FILE"

    # ----------------------- 1. 构建环境镜像 -----------------------
    echo "[步骤 1/3] 构建/检查 Docker 环境镜像..."
    echo "--- 1. 构建环境镜像 ---" >> "$LOG_FILE"
    
    python -m swebench.harness.prepare_images \
        --dataset_name SWE-bench/SWE-bench_Lite \
        --split test \
        --instance_ids "$INSTANCE_ID" \
        --max_workers 1 \
        --tag latest \
        --env_image_tag latest \
        >> "$LOG_FILE" 2>&1 || {
            echo "❌ [ERROR] 镜像构建失败，跳过 $INSTANCE_ID"
            echo "错误：镜像构建失败，跳过 $INSTANCE_ID" >> "$LOG_FILE"
            docker container prune -f >> "$LOG_FILE" 2>&1
            docker image prune -a -f >> "$LOG_FILE" 2>&1
            echo "⏭️  继续处理下一个实例..."
            continue
        }
    
    echo "✅ [步骤 1/3] 环境镜像就绪"

    cd "$PROJECT_DIR"

    # ----------------------- 2. 运行增强版 -----------------------
    echo "[步骤 2/3] 运行增强版实验..."
    echo "--- 2. 运行增强版实验 ---" >> "$LOG_FILE"
    
    python -m swebench.experiments.enhanced_patch_pipeline \
        --dataset_name SWE-bench/SWE-bench_Lite \
        --split test \
        --instance_ids "$INSTANCE_ID" \
        --model "$MODEL" \
        --output_dir "$OUTPUT_ENHANCED" \
        >> "$LOG_FILE" 2>&1
    
    echo "✅ [步骤 2/3] 增强版实验完成"
    date >> "$LOG_FILE"

    # ----------------------- 3. 运行基线版 -----------------------
    echo "[步骤 3/3] 运行基线版实验..."
    echo "--- 3. 运行基线版实验 ---" >> "$LOG_FILE"
    
    python -m swebench.experiments.enhanced_patch_pipeline \
        --dataset_name SWE-bench/SWE-bench_Lite \
        --split test \
        --instance_ids "$INSTANCE_ID" \
        --model "$MODEL" \
        --output_dir "$OUTPUT_BASELINE" \
        --baseline_only \
        --hide_original_test_patch_in_repair \
        >> "$LOG_FILE" 2>&1
    
    echo "✅ [步骤 3/3] 基线版实验完成"
    date >> "$LOG_FILE"

    # ----------------------- 4. 清理与统计 -----------------------
    INSTANCE_END=$(date +%s)
    INSTANCE_DURATION=$((INSTANCE_END - INSTANCE_START))
    echo "⏱️  当前实例耗时: ${INSTANCE_DURATION}秒"

    # 计算预估剩余时间
    TOTAL_ELAPSED=$((INSTANCE_END - START_TIME))
    AVG_TIME=$((TOTAL_ELAPSED / CURRENT))
    REMAINING=$(( (TOTAL - CURRENT) * AVG_TIME ))
    echo "📊 已运行: ${CURRENT}/${TOTAL} | 平均耗时: ${AVG_TIME}秒/实例 | 预计剩余: $((REMAINING / 60))分钟"

    echo "--- 4. 清理 Docker 资源 ---" >> "$LOG_FILE"
    docker ps -aq --filter "name=sweb" | xargs -r docker rm -f >> "$LOG_FILE" 2>&1
    docker container prune -f >> "$LOG_FILE" 2>&1
    docker image prune -a -f >> "$LOG_FILE" 2>&1
    docker builder prune -f >> "$LOG_FILE" 2>&1
    docker system prune -a -f --volumes >> "$LOG_FILE" 2>&1

    echo "💾 磁盘剩余空间:"
    df -h / | grep -v Filesystem | tee -a "$LOG_FILE"
    echo ""
done

echo ""
echo "=============================================="
echo "🎉 所有实例处理完毕！"
echo "📁 增强版结果: $OUTPUT_ENHANCED"
echo "📁 基线版结果: $OUTPUT_BASELINE"
echo "📁 日志目录: $LOG_DIR"
TOTAL_END=$(date +%s)
echo "⏱️  总耗时: $(( (TOTAL_END - START_TIME) / 60 ))分钟"
echo "=============================================="
