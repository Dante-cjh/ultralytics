#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# Balloon 数据集自适应尺寸推理测试脚本
# 自动根据图片尺寸调整到最近的32倍数进行推理
################################################################################

set -e  # 遇到错误立即退出

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ultralytics

# ============================================================================
# 配置参数
# ============================================================================

# 设备配置
DEVICE="cuda:5"

# 模型路径（可根据需要修改）
MODEL_PATH="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 数据路径
BALLOON_DATA_PATH="/home/cjh/mmdetection/data/balloon/yolo_format"
VAL_IMAGES="${BALLOON_DATA_PATH}/images/val"
TEST_IMAGE="${BALLOON_DATA_PATH}/images/val/14898532020_ba6199dd22_k.jpg"

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

# 分隔线
print_separator() {
    echo "=============================================================================="
}

# ============================================================================
# 主程序
# ============================================================================

print_separator
log_info "开始 Balloon 自适应尺寸推理测试"
print_separator

# 检查模型文件
if [ ! -f "${MODEL_PATH}" ]; then
    log_error "模型文件不存在: ${MODEL_PATH}"
    log_info "请修改 MODEL_PATH 变量指向正确的模型路径"
    exit 1
fi

log_info "使用模型: ${MODEL_PATH}"
log_info "置信度阈值: ${CONFIDENCE}"
log_info "IoU阈值: ${IOU_THRESHOLD}"
log_info "设备: ${DEVICE}"
echo ""

# ============================================================================
# 测试1: 单张图像推理
# ============================================================================

print_separator
log_info "测试1: 单张图像自适应推理"
print_separator

SAVE_DIR_1="runs/test_adaptive/single_image_${TIMESTAMP}"

log_info "输入图像: ${TEST_IMAGE}"
log_info "保存目录: ${SAVE_DIR_1}"
echo ""

python balloon_inference_adaptive.py \
    --model "${MODEL_PATH}" \
    --source "${TEST_IMAGE}" \
    --confidence ${CONFIDENCE} \
    --iou ${IOU_THRESHOLD} \
    --device ${DEVICE} \
    --save-dir "${SAVE_DIR_1}"

log_info "测试1 完成！"
echo ""

# ============================================================================
# 测试2: 批量目录推理
# ============================================================================

print_separator
log_info "测试2: 批量目录自适应推理"
print_separator

SAVE_DIR_2="runs/test_adaptive/batch_images_${TIMESTAMP}"

log_info "输入目录: ${VAL_IMAGES}"
log_info "保存目录: ${SAVE_DIR_2}"
echo ""

python balloon_inference_adaptive.py \
    --model "${MODEL_PATH}" \
    --source "${VAL_IMAGES}" \
    --confidence ${CONFIDENCE} \
    --iou ${IOU_THRESHOLD} \
    --device ${DEVICE} \
    --save-dir "${SAVE_DIR_2}"

log_info "测试2 完成！"
echo ""

# ============================================================================
# 测试3: 不同置信度阈值测试
# ============================================================================

print_separator
log_info "测试3: 不同置信度阈值测试"
print_separator

CONFIDENCE_VALUES=(0.1 0.3 0.5)

for conf in "${CONFIDENCE_VALUES[@]}"; do
    log_info "测试置信度: ${conf}"
    SAVE_DIR_3="runs/test_adaptive/conf_${conf}_${TIMESTAMP}"
    
    python balloon_inference_adaptive.py \
        --model "${MODEL_PATH}" \
        --source "${TEST_IMAGE}" \
        --confidence ${conf} \
        --iou ${IOU_THRESHOLD} \
        --device ${DEVICE} \
        --save-dir "${SAVE_DIR_3}"
    
    echo ""
done

log_info "测试3 完成！"
echo ""

# ============================================================================
# 测试4: 验证模式测试
# ============================================================================

print_separator
log_info "测试4: 验证模式测试"
print_separator

SAVE_DIR_4="runs/test_adaptive/validation_${TIMESTAMP}"
DATA_YAML="my_balloon.yaml"

log_info "数据配置: ${DATA_YAML}"
log_info "保存目录: ${SAVE_DIR_4}"
echo ""

python balloon_inference_adaptive.py \
    --model "${MODEL_PATH}" \
    --val \
    --data "${DATA_YAML}" \
    --confidence ${CONFIDENCE} \
    --iou ${IOU_THRESHOLD} \
    --device ${DEVICE} \
    --batch 16 \
    --imgsz 640 \
    --save-dir "${SAVE_DIR_4}" \
    --name "val"

log_info "测试4 完成！"
echo ""

# ============================================================================
# 测试完成总结
# ============================================================================

print_separator
log_info "所有测试完成！"
print_separator

log_info "测试结果保存在以下目录:"
log_info "  测试1 (单张图像): ${SAVE_DIR_1}"
log_info "  测试2 (批量目录): ${SAVE_DIR_2}"
log_info "  测试3 (不同置信度): runs/test_adaptive/conf_*_${TIMESTAMP}"
log_info "  测试4 (验证模式): ${SAVE_DIR_4}"

print_separator
log_info "自适应尺寸推理测试脚本执行完毕！"
print_separator

