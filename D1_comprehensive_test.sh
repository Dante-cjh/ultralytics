#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# D1 数据集综合测试脚本
# 整合三种推理方式：常规推理、SAHI切片推理、多尺度推理
# 用于新数据集的验收测试
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数 - 请根据实际情况修改
# ============================================================================

# 新数据集路径（存放待测试图像的文件夹）
NEW_DATA_DIR="/path/to/new/test/images"

# 总保存路径（所有结果都保存在这个目录下）
OUTPUT_ROOT="runs/comprehensive_test"

# 设备配置
DEVICE=0

# 模型路径配置
# 1. 常规模型（正常训练的模型）
NORMAL_MODEL="runs/detect/D1_yolo11l_20241211/weights/best.pt"

# 2. 切片模型（使用切片数据训练的模型）
SLICE_MODEL="runs/detect/D1_yolo11l_slice_20241211/weights/best.pt"

# 3. 多尺度模型（用于多尺度推理，可以和常规模型相同）
MULTISCALE_MODEL="runs/detect/D1_yolo11l_20241211/weights/best.pt"

# 推理参数
CONFIDENCE=0.3
IOU_THRESHOLD=0.5

# SAHI推理参数
SLICE_HEIGHT=640
SLICE_WIDTH=640
OVERLAP_RATIO=0.15
POSTPROCESS_TYPE="NMS"
POSTPROCESS_THRESHOLD=0.6
POSTPROCESS_METRIC="IOS"
MIN_BOX_AREA=200
MAX_DETECTIONS=50

# 多尺度推理参数
SCALES="640 832 1024 1280"
FUSION_METHOD="nms"

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

log_section() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

# 统计检测数量函数
count_detections() {
    local labels_dir=$1
    local method_name=$2
    
    log_info "📊 统计 ${method_name} 的检测结果..."
    
    if [ ! -d "${labels_dir}" ]; then
        log_error "标签目录不存在: ${labels_dir}"
        return 1
    fi
    
    local total_detections=0
    local image_count=0
    
    # 创建临时文件保存详细统计
    local detail_file="${labels_dir}/../detection_stats.txt"
    echo "=== ${method_name} 检测统计 ===" > "${detail_file}"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${detail_file}"
    echo "" >> "${detail_file}"
    
    # 遍历所有标签文件
    for label_file in "${labels_dir}"/*.txt; do
        if [ -f "${label_file}" ]; then
            local filename=$(basename "${label_file}")
            local count=$(wc -l < "${label_file}" 2>/dev/null || echo "0")
            
            echo "  ${filename}: ${count} 个目标" >> "${detail_file}"
            log_info "  ${filename}: ${count} 个目标"
            
            total_detections=$((total_detections + count))
            image_count=$((image_count + 1))
        fi
    done
    
    if [ ${image_count} -eq 0 ]; then
        log_error "未找到任何标签文件"
        return 1
    fi
    
    local avg_detections=$(echo "scale=2; ${total_detections} / ${image_count}" | bc)
    
    echo "" >> "${detail_file}"
    echo "=== 总结 ===" >> "${detail_file}"
    echo "图像总数: ${image_count}" >> "${detail_file}"
    echo "检测总数: ${total_detections}" >> "${detail_file}"
    echo "平均每张: ${avg_detections} 个目标" >> "${detail_file}"
    
    log_section "📊 ${method_name} 统计结果"
    log_info "  图像总数: ${image_count}"
    log_info "  检测总数: ${total_detections}"
    log_info "  平均每张: ${avg_detections} 个目标"
    log_info "  详细统计: ${detail_file}"
    
    return 0
}

# 1. 常规推理函数
run_normal_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法1: 常规推理"
    log_info "模型: ${model_path}"
    log_info "数据: ${source_dir}"
    log_info "保存: ${save_dir}"
    
    if [ ! -f "${model_path}" ]; then
        log_error "模型文件不存在: ${model_path}"
        return 1
    fi
    
    if [ ! -d "${source_dir}" ]; then
        log_error "数据目录不存在: ${source_dir}"
        return 1
    fi
    
    python3 balloon_inference.py \
        --model "${model_path}" \
        --source "${source_dir}" \
        --save-dir "${save_dir}" \
        --confidence ${CONFIDENCE} \
        --iou ${IOU_THRESHOLD} \
        --device "cuda:${DEVICE}"
    
    if [ $? -eq 0 ]; then
        log_info "✅ 常规推理完成"
        count_detections "${save_dir}/labels" "常规推理"
    else
        log_error "❌ 常规推理失败"
        return 1
    fi
}

# 2. SAHI切片推理函数
run_sahi_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法2: SAHI切片推理"
    log_info "模型: ${model_path}"
    log_info "数据: ${source_dir}"
    log_info "保存: ${save_dir}"
    
    if [ ! -f "${model_path}" ]; then
        log_error "模型文件不存在: ${model_path}"
        return 1
    fi
    
    if [ ! -d "${source_dir}" ]; then
        log_error "数据目录不存在: ${source_dir}"
        return 1
    fi
    
    python3 D1_inference_with_sahi_v3.py \
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
        log_info "✅ SAHI切片推理完成"
        count_detections "${save_dir}/labels" "SAHI切片推理"
    else
        log_error "❌ SAHI切片推理失败"
        return 1
    fi
}

# 3. 多尺度推理函数
run_multiscale_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法3: 多尺度推理"
    log_info "模型: ${model_path}"
    log_info "数据: ${source_dir}"
    log_info "保存: ${save_dir}"
    log_info "尺度: ${SCALES}"
    
    if [ ! -f "${model_path}" ]; then
        log_error "模型文件不存在: ${model_path}"
        return 1
    fi
    
    if [ ! -d "${source_dir}" ]; then
        log_error "数据目录不存在: ${source_dir}"
        return 1
    fi
    
    python3 balloon_inference_multiscale.py \
        --model "${model_path}" \
        --source "${source_dir}" \
        --save-dir "${save_dir}" \
        --scales ${SCALES} \
        --confidence ${CONFIDENCE} \
        --iou ${IOU_THRESHOLD} \
        --device "cuda:${DEVICE}" \
        --fusion ${FUSION_METHOD}
    
    if [ $? -eq 0 ]; then
        log_info "✅ 多尺度推理完成"
        count_detections "${save_dir}/labels" "多尺度推理"
    else
        log_error "❌ 多尺度推理失败"
        return 1
    fi
}

# ============================================================================
# 主流程
# ============================================================================

log_section "🚀 D1 数据集综合测试开始"
log_info "测试时间: ${TIMESTAMP}"
log_info "新数据集: ${NEW_DATA_DIR}"
log_info "保存路径: ${OUTPUT_ROOT}"

# 验证输入
if [ ! -d "${NEW_DATA_DIR}" ]; then
    log_error "新数据集目录不存在: ${NEW_DATA_DIR}"
    log_error "请修改脚本中的 NEW_DATA_DIR 变量"
    exit 1
fi

# 统计图像数量
IMAGE_COUNT=$(find "${NEW_DATA_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
log_info "待测试图像: ${IMAGE_COUNT} 张"

if [ ${IMAGE_COUNT} -eq 0 ]; then
    log_error "未找到任何图像文件"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_ROOT}"

# 记录开始时间
START_TIME=$(date +%s)

# ============================================================================
# 方法1: 常规推理
# ============================================================================

if [ -f "${NORMAL_MODEL}" ]; then
    NORMAL_SAVE_DIR="${OUTPUT_ROOT}/01_normal_inference_${TIMESTAMP}"
    run_normal_inference "${NORMAL_MODEL}" "${NEW_DATA_DIR}" "${NORMAL_SAVE_DIR}" || log_error "常规推理失败"
else
    log_error "常规模型不存在: ${NORMAL_MODEL}"
    log_error "请修改脚本中的 NORMAL_MODEL 变量"
fi

# ============================================================================
# 方法2: SAHI切片推理
# ============================================================================

if [ -f "${SLICE_MODEL}" ]; then
    SAHI_SAVE_DIR="${OUTPUT_ROOT}/02_sahi_inference_${TIMESTAMP}"
    run_sahi_inference "${SLICE_MODEL}" "${NEW_DATA_DIR}" "${SAHI_SAVE_DIR}" || log_error "SAHI推理失败"
else
    log_error "切片模型不存在: ${SLICE_MODEL}"
    log_error "请修改脚本中的 SLICE_MODEL 变量"
fi

# ============================================================================
# 方法3: 多尺度推理
# ============================================================================

if [ -f "${MULTISCALE_MODEL}" ]; then
    MULTISCALE_SAVE_DIR="${OUTPUT_ROOT}/03_multiscale_inference_${TIMESTAMP}"
    run_multiscale_inference "${MULTISCALE_MODEL}" "${NEW_DATA_DIR}" "${MULTISCALE_SAVE_DIR}" || log_error "多尺度推理失败"
else
    log_error "多尺度模型不存在: ${MULTISCALE_MODEL}"
    log_error "请修改脚本中的 MULTISCALE_MODEL 变量"
fi

# ============================================================================
# 总结
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_section "🎉 综合测试完成"
log_info "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log_info ""
log_info "📁 结果保存位置:"
log_info "  总目录: ${OUTPUT_ROOT}"
log_info ""

if [ -d "${NORMAL_SAVE_DIR}" ]; then
    log_info "  1️⃣  常规推理:"
    log_info "     图像: ${NORMAL_SAVE_DIR}/*_visual.jpg"
    log_info "     标签: ${NORMAL_SAVE_DIR}/labels/*.txt"
    log_info "     统计: ${NORMAL_SAVE_DIR}/detection_stats.txt"
    log_info ""
fi

if [ -d "${SAHI_SAVE_DIR}" ]; then
    log_info "  2️⃣  SAHI切片推理:"
    log_info "     图像: ${SAHI_SAVE_DIR}/*_visual.jpg"
    log_info "     标签: ${SAHI_SAVE_DIR}/labels/*.txt"
    log_info "     统计: ${SAHI_SAVE_DIR}/detection_stats.txt"
    log_info ""
fi

if [ -d "${MULTISCALE_SAVE_DIR}" ]; then
    log_info "  3️⃣  多尺度推理:"
    log_info "     图像: ${MULTISCALE_SAVE_DIR}/*_multiscale.jpg"
    log_info "     标签: ${MULTISCALE_SAVE_DIR}/labels/*.txt"
    log_info "     统计: ${MULTISCALE_SAVE_DIR}/detection_stats.txt"
    log_info ""
fi

log_info "========================================================================"
log_info ""
log_info "💡 使用说明："
log_info "  1. 查看可视化结果图片，了解检测效果"
log_info "  2. 查看 detection_stats.txt 文件，了解每张图片的检测数量"
log_info "  3. 比较三种方法的检测结果，选择最优方案"
log_info ""
log_info "📊 快速对比命令："
log_info "  cat ${OUTPUT_ROOT}/*/detection_stats.txt"
log_info ""

