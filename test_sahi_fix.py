#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试SAHI推理修复效果 - 比较切片和合并结果
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi import AutoDetectionModel

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from balloon_inference_with_sahi import BalloonSAHIInference
from ultralytics.utils import LOGGER

def visualize_slices(image, detection_model, slice_height=640, slice_width=640, overlap_ratio=0.2):
    """可视化切片过程"""
    h, w = image.shape[:2]
    LOGGER.info(f"📐 图像尺寸: {w}x{h}")
    LOGGER.info(f"🔪 切片参数: {slice_width}x{slice_height}, 重叠: {overlap_ratio:.1%}")
    
    # 计算切片数量
    step_h = int(slice_height * (1 - overlap_ratio))
    step_w = int(slice_width * (1 - overlap_ratio))
    
    num_slices_h = max(1, (h - slice_height) // step_h + 1)
    num_slices_w = max(1, (w - slice_width) // step_w + 1)
    
    LOGGER.info(f"📊 切片数量: {num_slices_w} x {num_slices_h} = {num_slices_w * num_slices_h} 个切片")
    
    # 创建切片可视化图像
    slice_vis = image.copy()
    
    # 绘制切片网格
    for i in range(num_slices_h):
        for j in range(num_slices_w):
            y1 = i * step_h
            x1 = j * step_w
            y2 = min(y1 + slice_height, h)
            x2 = min(x1 + slice_width, w)
            
            # 绘制切片边界（红色）
            cv2.rectangle(slice_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 添加切片编号
            cv2.putText(slice_vis, f"{i*num_slices_w + j + 1}", 
                       (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return slice_vis, num_slices_h, num_slices_w

def test_individual_slices(image, detection_model, slice_height=640, slice_width=640, overlap_ratio=0.2, save_dir="runs/sahi_inference/test_fix"):
    """测试每个切片的单独检测结果"""
    h, w = image.shape[:2]
    
    # 计算切片参数
    step_h = int(slice_height * (1 - overlap_ratio))
    step_w = int(slice_width * (1 - overlap_ratio))
    
    num_slices_h = max(1, (h - slice_height) // step_h + 1)
    num_slices_w = max(1, (w - slice_width) // step_w + 1)
    
    LOGGER.info(f"🔍 开始测试每个切片的检测结果...")
    
    # 创建切片结果目录
    slice_dir = Path(save_dir) / "individual_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    
    slice_results = []
    
    for i in range(num_slices_h):
        for j in range(num_slices_w):
            y1 = i * step_h
            x1 = j * step_w
            y2 = min(y1 + slice_height, h)
            x2 = min(x1 + slice_width, w)
            
            # 提取切片
            slice_img = image[y1:y2, x1:x2]
            
            # 对切片进行检测
            try:
                slice_result = get_sliced_prediction(
                    slice_img,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=0.0,  # 单个切片不需要重叠
                    overlap_width_ratio=0.0,
                )
                
                # 可视化切片检测结果
                vis_slice = slice_img.copy()
                for pred in slice_result.object_prediction_list:
                    bbox = pred.bbox.to_xyxy()
                    x1_det, y1_det, x2_det, y2_det = map(int, bbox)
                    
                    # 绘制检测框
                    cv2.rectangle(vis_slice, (x1_det, y1_det), (x2_det, y2_det), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{pred.category.name}: {pred.score.value:.2f}"
                    cv2.putText(vis_slice, label, (x1_det, y1_det - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 保存切片检测结果
                slice_filename = f"slice_{i*num_slices_w + j + 1:02d}_detection.jpg"
                slice_path = slice_dir / slice_filename
                cv2.imwrite(str(slice_path), cv2.cvtColor(vis_slice, cv2.COLOR_RGB2BGR))
                
                slice_results.append({
                    'slice_id': i * num_slices_w + j + 1,
                    'coordinates': (x1, y1, x2, y2),
                    'detections': len(slice_result.object_prediction_list),
                    'result': slice_result
                })
                
                LOGGER.info(f"   切片 {i*num_slices_w + j + 1}: 位置({x1},{y1})-({x2},{y2}), 检测到{len(slice_result.object_prediction_list)}个目标")
                
            except Exception as e:
                LOGGER.error(f"   切片 {i*num_slices_w + j + 1} 检测失败: {e}")
    
    LOGGER.info(f"📁 切片检测结果保存到: {slice_dir}")
    return slice_results

def test_sahi_inference():
    """测试SAHI推理 - 比较切片和合并结果"""
    
    # 测试参数
    model_path = "runs/detect/balloon_yolo11l_slice_20251023_102255/weights/best.pt"
    test_image = "/home/cjh/mmdetection/data/balloon/yolo_format/images/val/24631331976_defa3bb61f_k.jpg"
    save_dir = "runs/sahi_inference/test_fix"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        LOGGER.error(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not Path(test_image).exists():
        LOGGER.error(f"❌ 测试图像不存在: {test_image}")
        return False
    
    try:
        # 创建推理器
        LOGGER.info("🚀 开始测试SAHI推理修复...")
        inferencer = BalloonSAHIInference(
            model_path=model_path,
            confidence_threshold=0.25,
            device="cuda:5"
        )
        
        # 读取图像
        image = read_image(test_image)
        h, w = image.shape[:2]
        
        # 1. 可视化切片过程
        LOGGER.info("📸 步骤1: 可视化切片过程...")
        slice_vis, num_slices_h, num_slices_w = visualize_slices(
            image, inferencer.detection_model, 640, 640, 0.2
        )
        
        # 保存切片可视化
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        slice_path = Path(save_dir) / "slice_visualization.jpg"
        cv2.imwrite(str(slice_path), cv2.cvtColor(slice_vis, cv2.COLOR_RGB2BGR))
        LOGGER.info(f"   切片可视化保存到: {slice_path}")
        
        # 2. 测试每个切片的单独检测结果
        LOGGER.info("🔍 步骤2: 测试每个切片的单独检测结果...")
        slice_results = test_individual_slices(
            image, inferencer.detection_model, 640, 640, 0.2, save_dir
        )
        
        # 3. 执行SAHI推理（合并结果）
        LOGGER.info("🔍 步骤3: 执行SAHI推理...")
        result = inferencer.predict_image(
            image_path=test_image,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            save_dir=save_dir,
            visualize=True
        )
        
        # 4. 创建对比图像
        LOGGER.info("📊 步骤4: 创建对比图像...")
        comparison = np.hstack([
            slice_vis,  # 左侧：切片可视化
            cv2.imread(str(Path(save_dir) / f"{Path(test_image).stem}_visual.jpg"))  # 右侧：合并结果
        ])
        
        # 添加标题
        cv2.putText(comparison, "Slice Visualization", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "SAHI Merged Result", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存对比图像
        comparison_path = Path(save_dir) / "comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        LOGGER.info(f"   对比图像保存到: {comparison_path}")
        
        # 5. 统计和总结
        total_slice_detections = sum(sr['detections'] for sr in slice_results)
        LOGGER.info(f"✅ 测试完成！")
        LOGGER.info(f"   图像尺寸: {result['image_size']}")
        LOGGER.info(f"   切片数量: {num_slices_w} x {num_slices_h}")
        LOGGER.info(f"   切片检测总数: {total_slice_detections}")
        LOGGER.info(f"   SAHI合并检测数: {result['num_detections']}")
        LOGGER.info(f"   结果保存到: {save_dir}")
        LOGGER.info(f"   - 切片可视化: {slice_path}")
        LOGGER.info(f"   - 对比图像: {comparison_path}")
        LOGGER.info(f"   - 单独切片检测: {Path(save_dir) / 'individual_slices'}")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sahi_inference()
    if success:
        print("\n🎉 SAHI推理修复测试成功！")
    else:
        print("\n❌ SAHI推理修复测试失败！")
