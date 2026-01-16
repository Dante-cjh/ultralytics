#!/bin/bash

# 两阶段级联检测 - 批量评估脚本
# 使用方法: bash run_cascaded_eval.sh

# ==================== 配置参数 ====================

# YOLO模型路径（第一阶段）
YOLO_MODEL="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 数据集配置
DATA_YAML="/home/cjh/ultralytics/my_balloon.yaml"

# 分类器参数
CLASSIFIER_TYPE="mobilenet"  # 分类器类型: mlp 或 mobilenet
INPUT_SIZE=112               # 分类器输入尺寸
NUM_CLASSES=2                # 类别数：1个前景类 + 1个背景类 = 2

# 分类器权重路径
# 可以通过命令行参数指定，或者自动查找最新的模型
# 格式: runs/mobilenet/<yolo_name>_<timestamp>/best.pt
CLASSIFIER_WEIGHT=""  # 留空则自动查找最新模型

# 评估参数
SPLIT="val"                 # 评估集: train 或 val
STAGE1_CONF=0.05           # 第一阶段置信度阈值
STAGE2_THRESHOLD=0.5       # 第二阶段分类阈值
YOLO_CONF=0.25             # 单阶段YOLO置信度（用于对比）
IMGSZ=1280                 # 推理尺寸

# 设备
DEVICE="cuda:0"

# ==================== 运行评估 ====================

echo "========================================="
echo "两阶段级联检测 - 批量评估"
echo "========================================="

# 提取YOLO模型名称
YOLO_MODEL_NAME=$(basename $(dirname $(dirname "$YOLO_MODEL")))

# 输出目录（类似runs/inference/<model_name>_val的结构）
EVAL_DIR="runs/inference/${YOLO_MODEL_NAME}_cascaded_${SPLIT}"

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

echo "YOLO模型: $YOLO_MODEL"
echo "分类器: $CLASSIFIER_WEIGHT"
echo "评估集: $SPLIT"
echo "第一阶段置信度: $STAGE1_CONF"
echo "第二阶段阈值: $STAGE2_THRESHOLD"
echo "对比YOLO置信度: $YOLO_CONF"
echo ""

python balloon_cascaded_infer_all.py \
    --yolo-model "$YOLO_MODEL" \
    --classifier "$CLASSIFIER_WEIGHT" \
    --data-yaml "$DATA_YAML" \
    --split "$SPLIT" \
    --model-type "$CLASSIFIER_TYPE" \
    --input-size $INPUT_SIZE \
    --num-classes $NUM_CLASSES \
    --imgsz $IMGSZ \
    --stage1-conf $STAGE1_CONF \
    --stage2-threshold $STAGE2_THRESHOLD \
    --yolo-conf $YOLO_CONF \
    --save-dir "$EVAL_DIR" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 批量评估完成!"
    echo "========================================="
    echo "详细结果: $EVAL_DIR/detailed_results.json"
    echo "评估报告: $EVAL_DIR/evaluation_report.txt"
    echo "可视化图像: $EVAL_DIR/visualizations/"
    echo "推理标签: $EVAL_DIR/labels_two_stage/"
    echo ""
    echo "查看报告:"
    echo "  cat $EVAL_DIR/evaluation_report.txt"
else
    echo "❌ 批量评估失败!"
    exit 1
fi

