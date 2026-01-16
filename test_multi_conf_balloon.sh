#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# 多置信度推理测试脚本
# 测试不同置信度阈值对检测数量准确率的影响
################################################################################

set -e

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ultralytics

cd /home/cjh/ultralytics

# ============================================================================
# 配置参数 - 请根据实际情况修改
# ============================================================================

# 模型路径 (修改为你的最佳模型)
MODEL="/home/cjh/ultralytics/runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 验证集路径
VAL_DIR="/home/cjh/mmdetection/data/balloon/yolo_format/images/val"
TRUE_LABELS_DIR="/home/cjh/mmdetection/data/balloon/yolo_format/labels/val"

# 设备
DEVICE="cuda:0"

# 推理参数
IMGSZ=1280
IOU=0.5

# 置信度列表
CONF_LIST="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4"

# 保存目录
SAVE_DIR="runs/multi_conf_test"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================================================
# 主流程
# ============================================================================

echo "========================================"
echo "多置信度推理测试"
echo "========================================"
echo "模型: ${MODEL}"
echo "图像目录: ${VAL_DIR}"
echo "标签目录: ${TRUE_LABELS_DIR}"
echo "置信度列表: ${CONF_LIST}"
echo "IoU阈值: ${IOU}"
echo "图像尺寸: ${IMGSZ}"
echo "========================================"

python3 balloon_inference_multi_conf.py \
    --model "${MODEL}" \
    --source "${VAL_DIR}" \
    --true-labels "${TRUE_LABELS_DIR}" \
    --save-dir "${SAVE_DIR}/${TIMESTAMP}" \
    --conf-list ${CONF_LIST} \
    --iou ${IOU} \
    --imgsz ${IMGSZ} \
    --device "${DEVICE}"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
echo "结果位置: ${SAVE_DIR}/${TIMESTAMP}/"
echo "详细结果: ${SAVE_DIR}/${TIMESTAMP}/multi_conf_results.json"

