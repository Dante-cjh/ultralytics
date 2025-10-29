#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# Balloon 数据集 - 单尺度切片多模型训练脚本
# 依次训练 YOLO11m, YOLO11l, YOLO11x 模型（使用切片数据）
# 第一次运行时进行数据切片，后续模型仅训练
# 
# 重要说明：
# - 训练使用切片数据（SLICE_DIR）提高小目标检测效果
# - SAHI推理使用原始完整图像（DATA_ROOT）进行切片推理后拼接
# - 这样可以得到完整图像的检测结果，而不是切片图像的检测结果
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 设备配置
DEVICE=5

# 数据路径
DATA_ROOT="/home/cjh/mmdetection/data/balloon/yolo_format"
SLICE_DIR="/home/cjh/mmdetection/data/balloon/yolo_format_slice"
# SAHI推理使用原始完整图像，而不是切片图像
VAL_DIR="${DATA_ROOT}/images/val"
TEST_DIR="${DATA_ROOT}/images/test"

# 切片参数
CROP_SIZE=640
GAP=100
RATES="1.0"

# 训练参数
EPOCHS=2
BATCH=16
PATIENCE=20

# SAHI推理参数
CONFIDENCE=0.3
SLICE_HEIGHT=640
SLICE_WIDTH=640
OVERLAP_RATIO=0.15
POSTPROCESS_TYPE="NMS"
POSTPROCESS_THRESHOLD=0.6
POSTPROCESS_METRIC="IOS"
MIN_BOX_AREA=200
MAX_DETECTIONS=50

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

# 数据切片函数（仅第一次执行）
slice_data() {
    log_info "=========================================="
    log_info "开始数据切片"
    log_info "原始数据: ${DATA_ROOT}"
    log_info "切片数据: ${SLICE_DIR}"
    log_info "=========================================="
    
    python3 balloon_training_with_slice.py \
        --data-root "${DATA_ROOT}" \
        --slice-dir "${SLICE_DIR}" \
        --crop-size ${CROP_SIZE} \
        --gap ${GAP} \
        --rates ${RATES} \
        --slice-only
    
    if [ $? -eq 0 ]; then
        log_info "✅ 数据切片完成"
    else
        log_error "❌ 数据切片失败"
        return 1
    fi
}

# 训练函数
train_model() {
    local model_name=$1
    local project_name=$2
    local train_only=$3
    
    log_info "=========================================="
    log_info "开始训练模型: ${model_name}"
    log_info "项目名称: ${project_name}"
    log_info "仅训练模式: ${train_only}"
    log_info "=========================================="
    
    if [ "${train_only}" == "true" ]; then
        # 仅训练，不切片
        python3 balloon_training_with_slice.py \
            --data-root "${DATA_ROOT}" \
            --slice-dir "${SLICE_DIR}" \
            --model "${model_name}" \
            --project-name "${project_name}" \
            --epochs ${EPOCHS} \
            --batch ${BATCH} \
            --device ${DEVICE} \
            --patience ${PATIENCE} \
            --train-only
    else
        # 完整流程（包括切片）
        python3 balloon_training_with_slice.py \
            --data-root "${DATA_ROOT}" \
            --slice-dir "${SLICE_DIR}" \
            --model "${model_name}" \
            --project-name "${project_name}" \
            --epochs ${EPOCHS} \
            --batch ${BATCH} \
            --device ${DEVICE} \
            --patience ${PATIENCE}
    fi
    
    if [ $? -eq 0 ]; then
        log_info "✅ 模型训练完成: ${model_name}"
    else
        log_error "❌ 模型训练失败: ${model_name}"
        return 1
    fi
}

# SAHI推理函数
run_sahi_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    local dataset_type=$4
    
    log_info "=========================================="
    log_info "开始SAHI推理 - ${dataset_type}"
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
    
    python3 balloon_inference_with_sahi.py \
        --model "${model_path}" \
        --source "${source_dir}" \
        --save-dir "${save_dir}" \
        --confidence ${CONFIDENCE} \
        --device "cuda:${DEVICE}" \
        --slice-height ${SLICE_HEIGHT} \
        --slice-width ${SLICE_WIDTH} \
        --overlap-height ${OVERLAP_RATIO} \
        --overlap-width ${OVERLAP_RATIO} \
        --postprocess-type ${POSTPROCESS_TYPE} \
        --postprocess-threshold ${POSTPROCESS_THRESHOLD} \
        --postprocess-metric ${POSTPROCESS_METRIC} \
        --min-box-area ${MIN_BOX_AREA} \
        --max-detections ${MAX_DETECTIONS}
    
    if [ $? -eq 0 ]; then
        log_info "✅ SAHI推理完成: ${dataset_type}"
    else
        log_error "❌ SAHI推理失败: ${dataset_type}"
        return 1
    fi
}

# ============================================================================
# 主流程
# ============================================================================

log_info "🚀 开始 Balloon 单尺度切片多模型训练流水线"
log_info "时间戳: ${TIMESTAMP}"
log_info "切片参数: crop_size=${CROP_SIZE}, gap=${GAP}, rates=${RATES}"
log_info "训练参数: epochs=${EPOCHS}, batch=${BATCH}, patience=${PATIENCE}, device=${DEVICE}"

# 定义模型列表
declare -a MODELS=("yolo11n.pt" "yolo11l.pt" "yolo11l.pt")
declare -a MODEL_NAMES=("yolo11m" "yolo11l" "yolo11x")

# 记录开始时间
START_TIME=$(date +%s)

# 训练所有模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_slice_${TIMESTAMP}"
    
    log_info ""
    log_info "======================================================================"
    log_info "[$((i+1))/${#MODELS[@]}] 处理模型: ${MODEL}"
    log_info "======================================================================"
    
    # 1. 训练模型（第一次不加--train-only，后续都加）
    if [ $i -eq 0 ]; then
        # 第一个模型：完整流程（包括切片）
        train_model "${MODEL}" "${PROJECT_NAME}" "false" || continue
    else
        # 后续模型：仅训练
        train_model "${MODEL}" "${PROJECT_NAME}" "true" || continue
    fi
    
    # 2. 获取最佳模型路径
    BEST_MODEL="runs/detect/${PROJECT_NAME}/weights/best.pt"
    
    if [ ! -f "${BEST_MODEL}" ]; then
        log_error "最佳模型不存在: ${BEST_MODEL}"
        continue
    fi
    
    # 3. 在验证集上进行SAHI推理（使用原始完整图像）
    VAL_SAVE_DIR="runs/sahi_inference/${PROJECT_NAME}_val"
    log_info "🔍 使用原始完整图像进行SAHI推理: ${VAL_DIR}"
    run_sahi_inference "${BEST_MODEL}" "${VAL_DIR}" "${VAL_SAVE_DIR}" "Validation" || log_error "验证集推理失败"
    
    # 4. 在测试集上进行SAHI推理（如果存在，使用原始完整图像）
    if [ -d "${TEST_DIR}" ]; then
        TEST_SAVE_DIR="runs/sahi_inference/${PROJECT_NAME}_test"
        log_info "🔍 使用原始完整图像进行SAHI推理: ${TEST_DIR}"
        run_sahi_inference "${BEST_MODEL}" "${TEST_DIR}" "${TEST_SAVE_DIR}" "Test" || log_error "测试集推理失败"
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
    PROJECT_NAME="balloon_${MODEL_NAME}_slice_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME}: runs/detect/${PROJECT_NAME}/"
done
log_info ""
log_info "🔍 SAHI推理结果位置:"
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="balloon_${MODEL_NAME}_slice_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME} (验证集): runs/sahi_inference/${PROJECT_NAME}_val/"
    if [ -d "${TEST_DIR}" ]; then
        log_info "  - ${MODEL_NAME} (测试集): runs/sahi_inference/${PROJECT_NAME}_test/"
    fi
done
log_info ""
log_info "📁 切片数据位置: ${SLICE_DIR}"
log_info ""
log_info "📈 查看TensorBoard日志:"
log_info "  tensorboard --logdir runs/detect/"
log_info "======================================================================"

