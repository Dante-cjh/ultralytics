#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# Balloon 数据集 - 多模型训练脚本
# 依次训练 YOLO11m, YOLO11l, YOLO11x 模型
# 每个模型训练完成后进行验证和SAHI推理
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 设备配置
DEVICE=1

# 训练参数
EPOCHS=2
BATCH=16
PATIENCE=20

# 数据路径
VAL_DIR="/home/cjh/mmdetection/data/balloon/yolo_format/images/val"
TEST_DIR="/home/cjh/mmdetection/data/balloon/yolo_format/images/test"

# 推理参数
CONFIDENCE=0.25
IOU_THRESHOLD=0.5

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================================================
# 函数定义
# ============================================================================

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# 训练函数
train_model() {
    local model_name=$1
    local project_name=$2
    
    log_info "=========================================="
    log_info "开始训练模型: ${model_name}"
    log_info "项目名称: ${project_name}"
    log_info "=========================================="
    
    python3 balloon_training.py \
        --model "${model_name}" \
        --project-name "${project_name}" \
        --epochs ${EPOCHS} \
        --batch ${BATCH} \
        --device ${DEVICE} \
        --patience ${PATIENCE} \
        --skip-export
    
    if [ $? -eq 0 ]; then
        log_info "✅ 模型训练完成: ${model_name}"
    else
        log_error "❌ 模型训练失败: ${model_name}"
        return 1
    fi
}

# 普通推理函数
run_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    local dataset_type=$4
    
    log_info "=========================================="
    log_info "开始推理 - ${dataset_type}"
    log_info "模型: ${model_path}"
    log_info "数据: ${source_dir}"
    log_info "=========================================="
    
    if [ ! -f "${model_path}" ]; then
        log_error "模型文件不存在: ${model_path}"
        return 1
    fi
    
    if [ ! -d "${source_dir}" ]; then
        log_error "数据目录不存在: ${source_dir}"
        return 1
    fi
    
    python3 balloon_inference.py \
        --model "${model_path}" \
        --source "${source_dir}" \
        --save-dir "${save_dir}" \
        --confidence ${CONFIDENCE} \
        --iou ${IOU_THRESHOLD} \
        --device "cuda:${DEVICE}"
    
    if [ $? -eq 0 ]; then
        log_info "✅ 推理完成: ${dataset_type}"
    else
        log_error "❌ 推理失败: ${dataset_type}"
        return 1
    fi
}

# ============================================================================
# 主流程
# ============================================================================

log_info "🚀 开始 Balloon 多模型训练流水线"
log_info "时间戳: ${TIMESTAMP}"
log_info "训练参数: epochs=${EPOCHS}, batch=${BATCH}, patience=${PATIENCE}, device=${DEVICE}"

# 定义模型列表
declare -a MODELS=("yolo11m.pt" "yolo11l.pt" "yolo11x.pt")
declare -a MODEL_NAMES=("yolo11m" "yolo11l" "yolo11x")

# 记录开始时间
START_TIME=$(date +%s)

# 训练所有模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_${TIMESTAMP}"
    
    log_info ""
    log_info "======================================================================"
    log_info "[$((i+1))/${#MODELS[@]}] 处理模型: ${MODEL}"
    log_info "======================================================================"
    
    # 1. 训练模型
    train_model "${MODEL}" "${PROJECT_NAME}" || continue
    
    # 2. 获取最佳模型路径
    BEST_MODEL="runs/detect/${PROJECT_NAME}/weights/best.pt"
    
    if [ ! -f "${BEST_MODEL}" ]; then
        log_error "最佳模型不存在: ${BEST_MODEL}"
        continue
    fi
    
    # 3. 在验证集上进行推理
    VAL_SAVE_DIR="runs/inference/${PROJECT_NAME}_val"
    run_inference "${BEST_MODEL}" "${VAL_DIR}" "${VAL_SAVE_DIR}" "Validation" || log_error "验证集推理失败"
    
    # 4. 在验证集上进行模型评估（生成验证图表）
    log_info "=========================================="
    log_info "开始模型评估 - ${PROJECT_NAME}"
    log_info "=========================================="
    
    python3 balloon_inference.py \
        --model "${BEST_MODEL}" \
        --data "/home/cjh/ultralytics/my_balloon.yaml" \
        --val \
        --batch 32 \
        --imgsz 640 \
        --confidence ${CONFIDENCE} \
        --iou ${IOU_THRESHOLD} \
        --device "cuda:${DEVICE}" \
        --save-dir "runs/val" \
        --name "${PROJECT_NAME}_val" || log_error "模型评估失败"
    
    # 5. 在测试集上进行推理（如果存在）
    if [ -d "${TEST_DIR}" ]; then
        TEST_SAVE_DIR="runs/inference/${PROJECT_NAME}_test"
        run_inference "${BEST_MODEL}" "${TEST_DIR}" "${TEST_SAVE_DIR}" "Test" || log_error "测试集推理失败"
    else
        log_info "⚠️ 测试集目录不存在，跳过测试集推理"
    fi
    
    log_info "✅ 模型 ${MODEL_NAME} 完整流程完成"
    log_info ""
done

# ============================================================================
# 总结
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_info "======================================================================"
log_info "🎉 所有模型训练完成！"
log_info "======================================================================"
log_info "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log_info ""
log_info "📊 训练结果位置:"
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME}: runs/detect/${PROJECT_NAME}/"
done
log_info ""
log_info "🔍 推理结果位置:"
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME} (验证集): runs/inference/${PROJECT_NAME}_val/"
    if [ -d "${TEST_DIR}" ]; then
        log_info "  - ${MODEL_NAME} (测试集): runs/inference/${PROJECT_NAME}_test/"
    fi
done
log_info ""
log_info "📊 验证评估结果位置:"
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME}: runs/val/${PROJECT_NAME}_val/"
done
log_info ""
log_info "📈 查看训练结果:"
log_info "  - 训练曲线图: runs/detect/{项目名}/results.png"
log_info "  - 混淆矩阵: runs/detect/{项目名}/confusion_matrix.png"
log_info "  - 详细结果: runs/detect/{项目名}/results.csv"
log_info "======================================================================"

