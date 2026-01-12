#!/bin/bash

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ultralytics

cd /home/cjh/ultralytics

MODEL="/home/cjh/ultralytics/runs/detect/balloon_test/weights/best.pt"
VAL_DIR="/home/cjh/mmdetection/data/balloon/yolo_format/images/val"
DEVICE="cuda:0"
CONFIDENCE=0.25
IOU=0.5

echo "========================================"
echo "测试1: 标准单尺度推理 (imgsz=640)"
echo "========================================"

python3 balloon_inference.py \
    --model "${MODEL}" \
    --source "${VAL_DIR}" \
    --imgsz 640 \
    --confidence ${CONFIDENCE} \
    --iou ${IOU} \
    --device "${DEVICE}" \
    --save-dir "runs/test_yolo11n/test1_standard_640"

echo ""
echo "========================================"
echo "测试2: 标准单尺度推理 (imgsz=1280)"
echo "========================================"

python3 balloon_inference.py \
    --model "${MODEL}" \
    --source "${VAL_DIR}" \
    --imgsz 1280 \
    --confidence ${CONFIDENCE} \
    --iou ${IOU} \
    --device "${DEVICE}" \
    --save-dir "runs/test_yolo11n/test2_standard_1280"

echo ""
echo "========================================"
echo "测试3: 多尺度推理 + NMS"
echo "========================================"

python3 balloon_inference_multiscale.py \
    --model "${MODEL}" \
    --source "${VAL_DIR}" \
    --scales 640 832 1024 1280 \
    --fusion nms \
    --confidence ${CONFIDENCE} \
    --iou ${IOU} \
    --device "${DEVICE}" \
    --save-dir "runs/test_yolo11n/test3_multiscale_nms"

echo ""
echo "========================================"
echo "测试4: 多尺度推理 + WBF"
echo "========================================"

python3 balloon_inference_multiscale.py \
    --model "${MODEL}" \
    --source "${VAL_DIR}" \
    --scales 640 832 1024 1280 \
    --fusion wbf \
    --confidence ${CONFIDENCE} \
    --iou 0.45 \
    --device "${DEVICE}" \
    --save-dir "runs/test_yolo11n/test4_multiscale_wbf"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"

