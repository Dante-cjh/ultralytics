#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# D1 数据集 - 单尺度切片多模型训练脚本
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
DEVICE=0

# 数据路径
DATA_ROOT="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format"
SLICE_DIR="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format_slice"
# SAHI推理使用原始完整图像，而不是切片图像
VAL_DIR="${DATA_ROOT}/images/val"
TEST_DIR="${DATA_ROOT}/images/test"

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
    
    python3 D1_inference_with_sahi_v2.py \
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

# 定义模型列表
declare -a MODELS=("yolov8l.pt")
declare -a MODEL_NAMES=("yolov8l")

# 记录开始时间
START_TIME=$(date +%s)

# 训练所有模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"

    # 注意：这里需要手动填写，只需要填写对应的文件夹名称
    PROJECT_NAME="D1_yolov8l_slice_20251029_174115"
    
    # 1. 获取最佳模型路径
    BEST_MODEL="runs/detect/${PROJECT_NAME}/weights/best.pt"
    
    if [ ! -f "${BEST_MODEL}" ]; then
        log_error "最佳模型不存在: ${BEST_MODEL}"
        continue
    fi
    
    # 2. 在验证集上进行SAHI推理（使用原始完整图像）
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

log_info "🔍 SAHI推理结果位置:"
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PROJECT_NAME="D1_${MODEL_NAME}_slice_${TIMESTAMP}"
    log_info "  - ${MODEL_NAME} (验证集): runs/sahi_inference/${PROJECT_NAME}_val/"
    if [ -d "${TEST_DIR}" ]; then
        log_info "  - ${MODEL_NAME} (测试集): runs/sahi_inference/${PROJECT_NAME}_test/"
    fi
done
log_info ""
log_info "======================================================================"

