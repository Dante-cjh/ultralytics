#!/bin/bash
# Balloon 完整训练+推理流水线脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "🎈 Balloon 完整训练+推理流水线"
echo "=================================="

# 激活环境
source /home/cjh/anaconda3/bin/activate ultralytics
cd /home/cjh/ultralytics

echo ""
echo "📋 配置参数:"
echo "  数据根目录: /home/cjh/mmdetection/data/balloon/yolo_format"
echo "  训练轮数: 50 epochs"
echo "  切片大小: 640x640"
echo "  模型: yolo11n.pt"
echo ""

# ============ 步骤 1: 数据切片 ============
echo "=================================="
echo "📸 步骤 1/3: 数据切片"
echo "=================================="

python balloon_training_with_slice.py \
    --slice-only \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_slice \
    --crop-size 640 \
    --gap 100 \
    --rates 1.0

echo ""
echo "✅ 数据切片完成！"
echo ""

# ============ 步骤 2: 模型训练 ============
echo "=================================="
echo "🚀 步骤 2/3: 模型训练 (50 epochs)"
echo "=================================="

python balloon_training_with_slice.py \
    --train-only \
    --model yolo11n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project-name balloon_demo

# 查找实际保存的模型路径
MODEL_DIR=$(find runs/detect -name "balloon_demo*" -type d | head -1)
if [ -z "$MODEL_DIR" ]; then
    echo "❌ 错误：未找到训练结果目录"
    exit 1
fi

BEST_MODEL="$MODEL_DIR/weights/best.pt"
LAST_MODEL="$MODEL_DIR/weights/last.pt"

if [ -f "$BEST_MODEL" ]; then
    MODEL_PATH="$BEST_MODEL"
    echo "✅ 使用最佳模型: $BEST_MODEL"
elif [ -f "$LAST_MODEL" ]; then
    MODEL_PATH="$LAST_MODEL"
    echo "⚠️  使用最后模型: $LAST_MODEL"
else
    echo "❌ 错误：未找到模型文件"
    exit 1
fi

echo ""
echo "✅ 模型训练完成！"
echo ""

# ============ 步骤 3: SAHI 推理 ============
echo "=================================="
echo "🔍 步骤 3/3: SAHI 切片推理"
echo "=================================="

# 推理验证集
python balloon_inference_with_sahi.py \
    --model "$MODEL_PATH" \
    --source /home/cjh/mmdetection/data/balloon/yolo_format/images/val/ \
    --slice-height 640 \
    --slice-width 640 \
    --overlap-height 0.2 \
    --overlap-width 0.2 \
    --save-dir runs/balloon_demo_inference \
    --confidence 0.25 \
    --device 0

echo ""
echo "✅ 推理完成！"
echo ""

# ============ 结果总结 ============
echo "=================================="
echo "🎉 完整流水线执行成功！"
echo "=================================="
echo ""
echo "📁 结果位置:"
echo "  • 切片数据: /home/cjh/mmdetection/data/balloon/yolo_format_slice/"
echo "  • 训练模型: $MODEL_PATH"
echo "  • 推理结果: runs/balloon_demo_inference/"
echo ""
echo "🖼️  查看推理结果:"
echo "  ls -lh runs/balloon_demo_inference/"
echo ""
echo "📊 查看训练曲线:"
echo "  tensorboard --logdir $MODEL_DIR"
echo ""

