#!/bin/bash

# 基于SAHI的两阶段级联检测 - 批量推理脚本
# 使用方法: bash run_sahi_cascaded_eval.sh

# ==================== 配置参数 ====================

# YOLO模型路径
YOLO_MODEL="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 数据集配置
DATA_YAML="/home/cjh/ultralytics/my_balloon.yaml"
SPLIT="val"  # 评估集: train 或 val

# 分类器参数
CLASSIFIER_TYPE="mobilenet"  # 分类器类型: mlp 或 mobilenet
INPUT_SIZE=112               # 分类器输入尺寸
NUM_CLASSES=2                # 类别数：1个前景类 + 1个背景类 = 2

# 分类器权重路径（留空则自动查找最新模型）
CLASSIFIER_WEIGHT=""

# SAHI参数
SLICE_HEIGHT=640             # 切片高度
SLICE_WIDTH=640              # 切片宽度
OVERLAP_RATIO=0.2            # 重叠比例（0.15-0.3推荐）
SAHI_CONF=0.25               # SAHI置信度阈值

# 二阶段参数
STAGE2_THRESHOLD=0.5         # 二阶段分类阈值

# 设备
DEVICE="cuda:0"

# ==================== 运行推理 ====================

echo "========================================="
echo "基于SAHI的两阶段级联检测 - 批量推理"
echo "========================================="

# 提取YOLO模型名称
YOLO_MODEL_NAME=$(basename $(dirname $(dirname "$YOLO_MODEL")))

# 输出目录
SAVE_DIR="runs/inference/${YOLO_MODEL_NAME}_sahi_cascaded_${SPLIT}"

# 确定分类器路径
if [ -n "$1" ]; then
    # 如果提供了模型目录名，使用指定的路径
    CLASSIFIER_WEIGHT="runs/mobilenet/$1/best.pt"
    echo "使用指定分类器: $1"
elif [ -z "$CLASSIFIER_WEIGHT" ]; then
    # 自动查找最新的模型
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
        ls -1dt runs/mobilenet/${YOLO_MODEL_NAME}_*/ 2>/dev/null | head -5 | while read dir; do
            echo "  $(basename $dir)"
        done
        exit 1
    fi
fi

# 检查模型文件是否存在
if [ ! -f "$CLASSIFIER_WEIGHT" ]; then
    echo "❌ 错误: 分类器文件不存在: $CLASSIFIER_WEIGHT"
    echo ""
    echo "可用的模型:"
    ls -1dt runs/mobilenet/${YOLO_MODEL_NAME}_*/ 2>/dev/null | head -5 | while read dir; do
        if [ -f "$dir/best.pt" ]; then
            echo "  ✓ $(basename $dir)"
        fi
    done
    exit 1
fi

# 检查YOLO模型
if [ ! -f "$YOLO_MODEL" ]; then
    echo "❌ 错误: YOLO模型不存在: $YOLO_MODEL"
    exit 1
fi

# 检查数据集配置
if [ ! -f "$DATA_YAML" ]; then
    echo "❌ 错误: 数据集配置不存在: $DATA_YAML"
    exit 1
fi

echo "YOLO模型: $YOLO_MODEL"
echo "分类器: $CLASSIFIER_WEIGHT"
echo "数据集: $DATA_YAML"
echo "评估集: $SPLIT"
echo "保存目录: $SAVE_DIR"
echo ""
echo "SAHI参数:"
echo "  - 切片尺寸: ${SLICE_HEIGHT}x${SLICE_WIDTH}"
echo "  - 重叠比例: ${OVERLAP_RATIO}"
echo "  - 置信度: ${SAHI_CONF}"
echo ""
echo "二阶段参数:"
echo "  - 分类阈值: ${STAGE2_THRESHOLD}"
echo ""

# 运行SAHI两阶段推理
python balloon_sahi_cascaded_infer_all.py \
    --yolo-model "$YOLO_MODEL" \
    --classifier "$CLASSIFIER_WEIGHT" \
    --data-yaml "$DATA_YAML" \
    --split "$SPLIT" \
    --model-type "$CLASSIFIER_TYPE" \
    --input-size $INPUT_SIZE \
    --num-classes $NUM_CLASSES \
    --slice-height $SLICE_HEIGHT \
    --slice-width $SLICE_WIDTH \
    --overlap-ratio $OVERLAP_RATIO \
    --sahi-conf $SAHI_CONF \
    --stage2-threshold $STAGE2_THRESHOLD \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SAHI两阶段推理完成!"
    echo "========================================="
    echo "推理图像: $SAVE_DIR/images/"
    echo "SAHI标签: $SAVE_DIR/labels_sahi/"
    echo "二阶段标签: $SAVE_DIR/labels_sahi_stage2/"
    echo "可视化对比: $SAVE_DIR/visualizations_comparison/"
    echo ""
    echo "详细结果: $SAVE_DIR/detailed_results.json"
    echo "评估报告: $SAVE_DIR/evaluation_report.txt"
    echo ""
    echo "查看报告:"
    echo "  cat $SAVE_DIR/evaluation_report.txt"
else
    echo "❌ SAHI两阶段推理失败!"
    exit 1
fi


