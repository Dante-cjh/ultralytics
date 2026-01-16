#!/bin/bash

# 两阶段级联检测 - 完整流程脚本
# 使用方法: bash run_cascaded_detection.sh [prepare|train|infer|all]

# ==================== 配置参数 ====================

# YOLO模型路径（第一阶段）
YOLO_MODEL="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 提取YOLO模型名称（从路径中提取）
# 例如: runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt -> balloon_yolo11l_20251203_160322
YOLO_MODEL_NAME=$(basename $(dirname $(dirname "$YOLO_MODEL")))

# 数据集配置
DATA_YAML="/home/cjh/ultralytics/my_balloon.yaml"

# 第一阶段参数
STAGE1_CONF=0.05        # 低置信度阈值，获取更多候选框
STAGE1_IOU=0.5          # 与GT匹配的IOU阈值
STAGE1_IMGSZ=1280       # 推理尺寸
FORCE_PREPARE=false     # 是否强制重新准备数据（true/false）

# 样本平衡参数
BALANCE_SAMPLES=true    # 是否平衡正负样本（true/false）
NEGATIVE_RATIO=2.0      # 负样本与正样本的比例（如2.0表示负:正=2:1）

# 分类器参数
CLASSIFIER_TYPE="mobilenet"  # 分类器类型: mlp 或 mobilenet
INPUT_SIZE=112               # 分类器输入尺寸
NUM_CLASSES=2                # 类别数：1个前景类 + 1个背景类 = 2

# 训练参数
BATCH_SIZE=32
EPOCHS=50
LR=0.001

# Loss函数参数
LOSS_TYPE="focal"       # 损失函数类型: ce (CrossEntropy) 或 focal (FocalLoss)
FOCAL_ALPHA=0.25        # Focal Loss的alpha参数（类别权重）
FOCAL_GAMMA=2.0         # Focal Loss的gamma参数（难易样本权重差异，推荐2.0-5.0）

# 第二阶段参数
STAGE2_THRESHOLD=0.5    # 分类器置信度阈值

# 输出目录（自动根据模型名称生成）
DATA_DIR="data/${YOLO_MODEL_NAME}_cascaded_data_balloon"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_DIR="runs/mobilenet/${YOLO_MODEL_NAME}_${TIMESTAMP}"
INFER_DIR="runs/cascaded_infer_balloon/${YOLO_MODEL_NAME}"

# 设备
DEVICE="cuda:0"

# 测试图像（用于推理演示）
TEST_IMAGE="/home/cjh/mmdetection/data/balloon/yolo_format/images/val/14898532020_ba6199dd22_k.jpg"

# ==================== 函数定义 ====================

prepare_data() {
    echo "========================================="
    echo "步骤1: 准备训练数据"
    echo "========================================="
    echo "YOLO模型: $YOLO_MODEL"
    echo "模型名称: $YOLO_MODEL_NAME"
    echo "第一阶段置信度: $STAGE1_CONF"
    echo "IOU阈值: $STAGE1_IOU"
    echo "输出目录: $DATA_DIR"
    echo "强制重新生成: $FORCE_PREPARE"
    echo ""
    
    # 构建命令
    CMD="python balloon_cascaded_detection.py prepare \
        --yolo-model \"$YOLO_MODEL\" \
        --data-yaml \"$DATA_YAML\" \
        --conf $STAGE1_CONF \
        --iou $STAGE1_IOU \
        --output-dir \"$DATA_DIR\" \
        --imgsz $STAGE1_IMGSZ \
        --negative-ratio $NEGATIVE_RATIO \
        --device \"$DEVICE\""
    
    # 如果强制重新生成，添加--force参数
    if [ "$FORCE_PREPARE" = "true" ]; then
        CMD="$CMD --force"
    fi
    
    # 如果不平衡样本，添加--no-balance参数
    if [ "$BALANCE_SAMPLES" = "false" ]; then
        CMD="$CMD --no-balance"
    fi
    
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 数据准备完成!"
    else
        echo "❌ 数据准备失败!"
        exit 1
    fi
}

train_classifier() {
    echo ""
    echo "========================================="
    echo "步骤2: 训练分类器"
    echo "========================================="
    echo "数据目录: $DATA_DIR"
    echo "模型类型: $CLASSIFIER_TYPE"
    echo "输入尺寸: ${INPUT_SIZE}x${INPUT_SIZE}"
    echo "类别数: $NUM_CLASSES"
    echo "批大小: $BATCH_SIZE"
    echo "训练轮数: $EPOCHS"
    echo "学习率: $LR"
    echo "损失函数: $LOSS_TYPE"
    if [ "$LOSS_TYPE" = "focal" ]; then
        echo "  - Focal Alpha: $FOCAL_ALPHA"
        echo "  - Focal Gamma: $FOCAL_GAMMA"
    fi
    echo ""
    
    python balloon_cascaded_detection.py train \
        --data-dir "$DATA_DIR" \
        --model-type "$CLASSIFIER_TYPE" \
        --input-size $INPUT_SIZE \
        --num-classes $NUM_CLASSES \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --save-dir "$TRAIN_DIR" \
        --device "$DEVICE" \
        --loss-type "$LOSS_TYPE" \
        --focal-alpha $FOCAL_ALPHA \
        --focal-gamma $FOCAL_GAMMA
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 分类器训练完成!"
    else
        echo "❌ 分类器训练失败!"
        exit 1
    fi
}

run_inference() {
    echo ""
    echo "========================================="
    echo "步骤3: 两阶段推理"
    echo "========================================="
    
    # 确定分类器路径
    local CLASSIFIER_PATH
    
    if [ -n "$1" ]; then
        # 如果提供了模型目录名，使用指定的路径
        CLASSIFIER_PATH="runs/mobilenet/$1/best.pt"
        echo "使用指定分类器: $1"
    else
        # 自动查找最新的模型
        local LATEST_MODEL=$(ls -t runs/mobilenet/${YOLO_MODEL_NAME}_*/best.pt 2>/dev/null | head -1)
        
        if [ -n "$LATEST_MODEL" ]; then
            CLASSIFIER_PATH="$LATEST_MODEL"
            local MODEL_DIR=$(basename $(dirname "$LATEST_MODEL"))
            echo "自动使用最新分类器: $MODEL_DIR"
        else
            echo "❌ 错误: 未找到训练好的分类器!"
            echo ""
            echo "请先训练分类器或手动指定模型目录:"
            echo "  bash run_cascaded_detection.sh train"
            echo "或者:"
            echo "  bash run_cascaded_detection.sh infer <model_dir_name>"
            echo ""
            echo "可用的模型:"
            ls -1dt runs/mobilenet/${YOLO_MODEL_NAME}_*/ 2>/dev/null | head -5 | while read dir; do
                echo "  $(basename $dir)"
            done
            exit 1
        fi
    fi
    
    # 检查模型文件是否存在
    if [ ! -f "$CLASSIFIER_PATH" ]; then
        echo "❌ 错误: 分类器文件不存在: $CLASSIFIER_PATH"
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
    echo "分类器: $CLASSIFIER_PATH"
    echo "测试图像: $TEST_IMAGE"
    echo ""
    
    python balloon_cascaded_detection.py infer \
        --yolo-model "$YOLO_MODEL" \
        --classifier "$CLASSIFIER_PATH" \
        --model-type "$CLASSIFIER_TYPE" \
        --image "$TEST_IMAGE" \
        --imgsz $STAGE1_IMGSZ \
        --input-size $INPUT_SIZE \
        --num-classes $NUM_CLASSES \
        --conf $STAGE1_CONF \
        --cls-threshold $STAGE2_THRESHOLD \
        --save-dir "$INFER_DIR" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 推理完成!"
    else
        echo "❌ 推理失败!"
        exit 1
    fi
}

# ==================== 主流程 ====================

case "$1" in
    prepare)
        prepare_data
        ;;
    train)
        train_classifier
        ;;
    infer)
        # 支持传入第二个参数指定模型目录
        run_inference "$2"
        ;;
    all)
        echo "开始完整的两阶段级联检测流程..."
        prepare_data
        train_classifier
        # 训练后直接使用新训练的模型
        CLASSIFIER_PATH="$TRAIN_DIR/best.pt"
        
        echo ""
        echo "========================================="
        echo "步骤3: 两阶段推理"
        echo "========================================="
        echo "YOLO模型: $YOLO_MODEL"
        echo "分类器: $CLASSIFIER_PATH"
        echo "测试图像: $TEST_IMAGE"
        echo ""
        
        python balloon_cascaded_detection.py infer \
            --yolo-model "$YOLO_MODEL" \
            --classifier "$CLASSIFIER_PATH" \
            --model-type "$CLASSIFIER_TYPE" \
            --image "$TEST_IMAGE" \
            --imgsz $STAGE1_IMGSZ \
            --input-size $INPUT_SIZE \
            --num-classes $NUM_CLASSES \
            --conf $STAGE1_CONF \
            --cls-threshold $STAGE2_THRESHOLD \
            --save-dir "$INFER_DIR" \
            --device "$DEVICE"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 推理完成!"
        else
            echo "❌ 推理失败!"
            exit 1
        fi
        
        echo ""
        echo "========================================="
        echo "✅ 全部流程完成!"
        echo "========================================="
        ;;
    *)
        echo "使用方法: bash $0 [prepare|train|infer|all] [model_dir]"
        echo ""
        echo "命令说明:"
        echo "  prepare          - 准备训练数据（使用YOLO生成候选框并标注）"
        echo "  train            - 训练分类器"
        echo "  infer [model]    - 运行两阶段推理"
        echo "                     可选参数: 指定模型目录名（不提供则自动使用最新）"
        echo "  all              - 运行完整流程（prepare + train + infer）"
        echo ""
        echo "示例:"
        echo "  bash $0 prepare                                              # 只准备数据"
        echo "  bash $0 train                                                # 只训练分类器"
        echo "  bash $0 infer                                                # 推理（自动使用最新模型）"
        echo "  bash $0 infer balloon_yolo11l_20251203_160322_20251209_124749  # 推理（指定模型）"
        echo "  bash $0 all                                                  # 运行完整流程"
        echo ""
        echo "查看可用的模型:"
        echo "  ls -lt runs/mobilenet/"
        exit 1
        ;;
esac

