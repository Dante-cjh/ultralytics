#!/bin/bash
# 快速检查切片数据质量的脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo "🔍 快速检查切片数据质量"
echo "======================================================"

# 默认数据路径
DATA_ROOT="${1:-/home/cjh/mmdetection/data/balloon/yolo_format_slice}"

if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}❌ 数据目录不存在: $DATA_ROOT${NC}"
    echo "用法: $0 [数据根目录]"
    exit 1
fi

echo "📁 数据根目录: $DATA_ROOT"
echo ""

# 检查 train 和 val
for split in train val; do
    echo "----------------------------------------"
    echo "🔍 检查 $split 数据集"
    echo "----------------------------------------"
    
    IMG_DIR="$DATA_ROOT/images/$split"
    LBL_DIR="$DATA_ROOT/labels/$split"
    
    if [ ! -d "$IMG_DIR" ]; then
        echo -e "${RED}❌ 图像目录不存在: $IMG_DIR${NC}"
        continue
    fi
    
    if [ ! -d "$LBL_DIR" ]; then
        echo -e "${RED}❌ 标签目录不存在: $LBL_DIR${NC}"
        continue
    fi
    
    # 统计图像和标签数量
    NUM_IMAGES=$(find "$IMG_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
    NUM_LABELS=$(find "$LBL_DIR" -name "*.txt" | wc -l)
    
    echo "图像数量: $NUM_IMAGES"
    echo "标签数量: $NUM_LABELS"
    
    # 检查是否匹配
    if [ $NUM_IMAGES -eq $NUM_LABELS ]; then
        echo -e "${GREEN}✅ 图像和标签数量匹配${NC}"
    else
        echo -e "${RED}❌ 图像和标签数量不匹配！${NC}"
        DIFF=$((NUM_IMAGES - NUM_LABELS))
        echo -e "${RED}   缺少 $DIFF 个标签文件${NC}"
    fi
    
    # 统计空标签文件
    NUM_EMPTY=0
    for label in "$LBL_DIR"/*.txt; do
        if [ -f "$label" ] && [ ! -s "$label" ]; then
            NUM_EMPTY=$((NUM_EMPTY + 1))
        fi
    done
    
    echo "空标签文件（负样本）: $NUM_EMPTY"
    
    if [ $NUM_EMPTY -eq 0 ] && [ $NUM_LABELS -gt 0 ]; then
        echo -e "${YELLOW}⚠️  警告: 没有负样本，可能导致爆框问题！${NC}"
    elif [ $NUM_EMPTY -gt 0 ]; then
        RATIO=$(awk "BEGIN {printf \"%.1f\", $NUM_EMPTY / $NUM_LABELS * 100}")
        echo -e "${GREEN}✅ 负样本占比: ${RATIO}%${NC}"
    fi
    
    # 检查缺失的标签
    echo ""
    echo "🔍 检查缺失的标签文件..."
    MISSING=0
    for img in "$IMG_DIR"/*.jpg "$IMG_DIR"/*.png; do
        if [ -f "$img" ]; then
            basename=$(basename "$img" | sed 's/\.[^.]*$//')
            label="$LBL_DIR/${basename}.txt"
            if [ ! -f "$label" ]; then
                MISSING=$((MISSING + 1))
                if [ $MISSING -le 5 ]; then
                    echo -e "${RED}  缺失: $(basename "$img")${NC}"
                fi
            fi
        fi
    done
    
    if [ $MISSING -gt 0 ]; then
        echo -e "${RED}❌ 总共缺失 $MISSING 个标签文件${NC}"
        echo -e "${YELLOW}💡 建议: 运行 check_slice_data_quality.py 进行详细检查${NC}"
    else
        echo -e "${GREEN}✅ 所有图像都有对应的标签文件${NC}"
    fi
    
    echo ""
done

echo "======================================================"
echo "✅ 快速检查完成"
echo "======================================================"
echo ""
echo "📋 详细检查命令:"
echo "python check_slice_data_quality.py \\"
echo "    --data-root $DATA_ROOT \\"
echo "    --splits train val \\"
echo "    --show-details \\"
echo "    --visualize"
echo ""

