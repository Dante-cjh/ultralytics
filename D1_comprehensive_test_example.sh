#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# D1 综合测试脚本 - 示例配置
# 这是一个预配置的示例，可以直接复制使用
################################################################################

set -e

# ============================================================================
# 快速配置区域（仅需修改这里）
# ============================================================================

# 新数据集路径 - 必须修改为实际路径
NEW_DATA_DIR="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_new_test/images"

# 输出根目录
OUTPUT_ROOT="runs/comprehensive_test_$(date +%Y%m%d)"

# GPU设备
DEVICE=0

# 三个模型路径（根据实际训练结果修改）
NORMAL_MODEL="runs/detect/D1_yolo11l_20241211_120000/weights/best.pt"
SLICE_MODEL="runs/detect/D1_yolo11l_slice_20241211_120000/weights/best.pt"
MULTISCALE_MODEL="runs/detect/D1_yolo11l_20241211_120000/weights/best.pt"

# ============================================================================
# 以下内容通常不需要修改
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIDENCE=0.3
IOU_THRESHOLD=0.5

# SAHI参数
SLICE_HEIGHT=640
SLICE_WIDTH=640
OVERLAP_RATIO=0.15
POSTPROCESS_TYPE="NMS"
POSTPROCESS_THRESHOLD=0.6
POSTPROCESS_METRIC="IOS"
MIN_BOX_AREA=200
MAX_DETECTIONS=50

# 多尺度参数
SCALES="640 832 1024 1280"
FUSION_METHOD="nms"

# ============================================================================
# 日志函数
# ============================================================================

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

# ============================================================================
# 统计函数
# ============================================================================

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
    local detail_file="${labels_dir}/../detection_stats.txt"
    
    echo "=== ${method_name} 检测统计 ===" > "${detail_file}"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${detail_file}"
    echo "" >> "${detail_file}"
    
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
}

# ============================================================================
# 推理函数
# ============================================================================

run_normal_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法1: 常规推理"
    
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

run_sahi_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法2: SAHI切片推理"
    
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

run_multiscale_inference() {
    local model_path=$1
    local source_dir=$2
    local save_dir=$3
    
    log_section "🔍 方法3: 多尺度推理"
    
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

log_section "🚀 D1 数据集综合测试"
log_info "测试时间: ${TIMESTAMP}"
log_info "新数据集: ${NEW_DATA_DIR}"
log_info "保存路径: ${OUTPUT_ROOT}"

# 验证输入
if [ ! -d "${NEW_DATA_DIR}" ]; then
    log_error "新数据集目录不存在: ${NEW_DATA_DIR}"
    exit 1
fi

# 统计图像
IMAGE_COUNT=$(find "${NEW_DATA_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
log_info "待测试图像: ${IMAGE_COUNT} 张"

if [ ${IMAGE_COUNT} -eq 0 ]; then
    log_error "未找到任何图像文件"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
START_TIME=$(date +%s)

# 方法1: 常规推理
if [ -f "${NORMAL_MODEL}" ]; then
    run_normal_inference "${NORMAL_MODEL}" "${NEW_DATA_DIR}" \
        "${OUTPUT_ROOT}/01_normal_inference_${TIMESTAMP}" || log_error "常规推理失败"
else
    log_error "常规模型不存在: ${NORMAL_MODEL}"
fi

# 方法2: SAHI推理
if [ -f "${SLICE_MODEL}" ]; then
    run_sahi_inference "${SLICE_MODEL}" "${NEW_DATA_DIR}" \
        "${OUTPUT_ROOT}/02_sahi_inference_${TIMESTAMP}" || log_error "SAHI推理失败"
else
    log_error "切片模型不存在: ${SLICE_MODEL}"
fi

# 方法3: 多尺度推理
if [ -f "${MULTISCALE_MODEL}" ]; then
    run_multiscale_inference "${MULTISCALE_MODEL}" "${NEW_DATA_DIR}" \
        "${OUTPUT_ROOT}/03_multiscale_inference_${TIMESTAMP}" || log_error "多尺度推理失败"
else
    log_error "多尺度模型不存在: ${MULTISCALE_MODEL}"
fi

# 总结
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_section "🎉 综合测试完成"
log_info "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log_info ""
log_info "📊 快速查看所有统计结果："
log_info "  cat ${OUTPUT_ROOT}/*/detection_stats.txt"
log_info ""

