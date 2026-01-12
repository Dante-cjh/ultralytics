#!/bin/bash

# SAHI结果的二阶段精修脚本
# 使用方法: bash run_cascaded_sahi.sh

# ==================== 配置参数 ====================

# SAHI推理结果目录（包含labels子目录）
# 例如: runs/sahi_inference/D1_yolov8l_slice_20251029_174115_val
SAHI_RESULTS_DIR="runs/sahi_inference/balloon_yolo11l_20251203_160322_val"

# 原始图像目录
IMAGES_DIR="data/balloon/images/val"

# 分类器路径
# 可以通过命令行参数指定，或者自动查找最新的模型
CLASSIFIER_WEIGHT=""  # 留空则自动查找最新模型

# 分类器参数
CLASSIFIER_TYPE="mobilenet"  # 分类器类型: mlp 或 mobilenet
INPUT_SIZE=112               # 分类器输入尺寸
NUM_CLASSES=2                # 类别数：1个前景类 + 1个背景类 = 2
THRESHOLD=0.5                # 分类阈值

# 输出目录
SAVE_DIR="runs/cascaded_sahi_refine"

# 设备
DEVICE="cuda:0"

# 类别名称（可选）
CLASS_NAMES="balloon"

# ==================== 运行精修 ====================

echo "========================================="
echo "SAHI结果的二阶段精修"
echo "========================================="

# 从SAHI结果目录提取YOLO模型名称
YOLO_MODEL_NAME=$(basename "$SAHI_RESULTS_DIR" | sed 's/_val$//' | sed 's/_test$//')

# 确定分类器路径
if [ -n "$1" ]; then
    # 如果提供了模型目录名，使用指定的路径
    CLASSIFIER_WEIGHT="runs/mobilenet/$1/best.pt"
    echo "使用指定分类器: $1"
elif [ -z "$CLASSIFIER_WEIGHT" ]; then
    # 自动查找最新的模型（匹配YOLO模型名称）
    LATEST_MODEL=$(ls -t runs/mobilenet/${YOLO_MODEL_NAME}_*/best.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_MODEL" ]; then
        CLASSIFIER_WEIGHT="$LATEST_MODEL"
        MODEL_DIR=$(basename $(dirname "$LATEST_MODEL"))
        echo "自动使用最新分类器: $MODEL_DIR"
    else
        echo "❌ 错误: 未找到训练好的分类器!"
        echo ""
        echo "请先训练分类器或手动指定模型目录:"
        echo "  bash run_cascaded_detection.sh train"
        echo "或者:"
        echo "  bash $0 <model_dir_name>"
        echo ""
        echo "可用的模型:"
        ls -1dt runs/mobilenet/*/ 2>/dev/null | head -5 | while read dir; do
            echo "  $(basename $dir)"
        done
        exit 1
    fi
fi

# 检查模型文件是否存在
if [ ! -f "$CLASSIFIER_WEIGHT" ]; then
    echo "❌ 错误: 分类器文件不存在: $CLASSIFIER_WEIGHT"
    exit 1
fi

# 检查SAHI结果目录
if [ ! -d "$SAHI_RESULTS_DIR/labels" ]; then
    echo "❌ 错误: SAHI结果目录不存在或没有labels子目录: $SAHI_RESULTS_DIR"
    echo ""
    echo "可用的SAHI结果:"
    ls -1dt runs/sahi_inference/*/ 2>/dev/null | head -5 | while read dir; do
        echo "  $(basename $dir)"
    done
    exit 1
fi

# 检查图像目录
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ 错误: 图像目录不存在: $IMAGES_DIR"
    exit 1
fi

echo "SAHI结果: $SAHI_RESULTS_DIR"
echo "图像目录: $IMAGES_DIR"
echo "分类器: $CLASSIFIER_WEIGHT"
echo "保存目录: $SAVE_DIR"
echo "分类阈值: $THRESHOLD"
echo ""

# 运行精修
python balloon_cascaded_from_sahi.py \
    --sahi-results "$SAHI_RESULTS_DIR" \
    --images "$IMAGES_DIR" \
    --classifier "$CLASSIFIER_WEIGHT" \
    --save-dir "$SAVE_DIR" \
    --model-type "$CLASSIFIER_TYPE" \
    --input-size $INPUT_SIZE \
    --num-classes $NUM_CLASSES \
    --threshold $THRESHOLD \
    --device "$DEVICE" \
    --class-names $CLASS_NAMES

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SAHI结果精修完成!"
    echo "========================================="
    echo "精修后labels: $SAVE_DIR/labels/"
    echo "可视化对比: $SAVE_DIR/visualizations/"
    echo ""
    echo "下一步:"
    echo "1. 查看可视化对比图: $SAVE_DIR/visualizations/"
    echo "2. 使用精修后的labels进行评估"
else
    echo "❌ SAHI结果精修失败!"
    exit 1
fi


