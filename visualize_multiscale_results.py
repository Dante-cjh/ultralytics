#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多尺度检测结果可视化脚本
将ground truth、每个尺度的检测结果、最终合并效果拼接到一张图上

使用方法:
python visualize_multiscale_results.py \
    --model best.pt \
    --image /path/to/image.jpg \
    --label /path/to/label.txt \
    --scales 640 832 1024 1280 \
    --save-dir runs/multiscale_visible
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def cross_class_nms(boxes_list, iou_threshold=0.5):
    """
    跨类别NMS：对所有类别的检测框进行NMS，去除重复检测
    用于解决多尺度融合时同一个目标被多个类别检测的问题
    
    Args:
        boxes_list: 检测框列表 [(cls, conf, x1, y1, x2, y2), ...]
        iou_threshold: IoU阈值
    
    Returns:
        过滤后的检测框列表
    """
    if len(boxes_list) == 0:
        return []
    
    # 提取所有检测框和置信度
    boxes = np.array([[b[2], b[3], b[4], b[5]] for b in boxes_list])  # x1,y1,x2,y2
    scores = np.array([b[1] for b in boxes_list])  # 置信度
    
    # 计算面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # 计算IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    # 返回保留的检测框
    return [boxes_list[i] for i in keep]


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """
    解析YOLO格式标签文件
    
    Args:
        label_path: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        LOGGER.warning(f"   ⚠️ 标签文件不存在: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            LOGGER.warning(f"   ⚠️ 标签文件为空: {label_path}")
            return boxes
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            parts = line.split()
            if len(parts) < 5:
                LOGGER.warning(f"   ⚠️ 标签格式错误 (行{line_num}): {line} - 需要至少5个值")
                continue
            
            try:
                cls = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 检查坐标是否合理（归一化值应该在0-1之间）
                if not (0 <= float(parts[1]) <= 1 and 0 <= float(parts[2]) <= 1 and 
                       0 <= float(parts[3]) <= 1 and 0 <= float(parts[4]) <= 1):
                    LOGGER.warning(f"   ⚠️ 坐标值超出范围 (行{line_num}): {line} - YOLO格式应该是归一化坐标(0-1)")
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append((cls, x1, y1, x2, y2))
            except ValueError as e:
                LOGGER.warning(f"   ⚠️ 解析错误 (行{line_num}): {line} - {e}")
                continue
    
    if len(boxes) > 0:
        LOGGER.info(f"   ✅ 成功解析标签: {label_path} ({len(boxes)} 个目标)")
    
    return boxes


def draw_boxes(image: np.ndarray, boxes: List, color: Tuple[int, int, int] = None, 
               label: str = "", thickness: int = 2, show_class: bool = True,
               class_names: dict = None, show_conf: bool = True) -> np.ndarray:
    """
    在图像上绘制检测框（按类别用不同颜色，显示类别和置信度）
    
    Args:
        image: 输入图像
        boxes: 检测框列表 [(cls, conf, x1, y1, x2, y2), ...] 或 [(cls, x1, y1, x2, y2), ...] (GT)
        color: 统一颜色 (如果为None则按类别自动分配)
        label: 标签文字
        thickness: 线宽
        show_class: 是否显示类别标签
        class_names: 类别名称字典 {0: 'class0', 1: 'class1', ...}
        show_conf: 是否显示置信度
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    # 类别颜色映射（支持多类别）
    class_colors = {
        0: (0, 255, 0),      # 绿色 - class 0
        1: (255, 0, 0),      # 蓝色 - class 1
        2: (0, 0, 255),      # 红色 - class 2
        3: (255, 255, 0),    # 青色 - class 3
        4: (255, 0, 255),    # 紫色 - class 4
        5: (0, 255, 255),    # 黄色 - class 5
    }
    
    # 默认类别名称
    if class_names is None:
        class_names = {0: 'hole', 1: 'cave', 2: 'unknow'}
    
    if len(boxes) == 0:
        LOGGER.debug(f"   draw_boxes: 没有框需要绘制")
    
    for box in boxes:
        if len(box) >= 5:
            # 判断是否包含置信度: (cls, conf, x1, y1, x2, y2) 或 (cls, x1, y1, x2, y2)
            if len(box) >= 6:
                cls, conf, x1, y1, x2, y2 = box[:6]
                conf = float(conf)
            else:
                cls, x1, y1, x2, y2 = box[:5]
                conf = None
            
            cls = int(cls)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 选择颜色：如果指定了color则用统一颜色，否则按类别分配
            box_color = color if color is not None else class_colors.get(cls, (128, 128, 128))
            
            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
            
            # 绘制类别和置信度标签
            if show_class or (show_conf and conf is not None):
                class_name = class_names.get(cls, f'cls{cls}')
                
                # 组合标签文字
                if conf is not None and show_conf:
                    label_text = f"{class_name} {conf:.2f}"
                else:
                    label_text = f"{class_name}"
                
                # 计算标签背景大小
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 标签位置
                label_y = max(y1 - 5, text_h + 5)
                
                # 绘制标签背景
                cv2.rectangle(
                    img,
                    (x1, label_y - text_h - baseline),
                    (x1 + text_w, label_y),
                    box_color,
                    -1
                )
                
                # 绘制标签文字（白色）
                cv2.putText(
                    img, label_text, (x1, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
    
    # 添加图片标题
    if label:
        cv2.putText(img, f"{label} ({len(boxes)})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img


def resize_with_padding(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    等比例缩放图像并填充到目标尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸
    
    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建目标大小的画布
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 居中放置
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def create_grid_image(images: List[np.ndarray], titles: List[str], 
                      cell_size: int = 640) -> np.ndarray:
    """
    将多张图像拼接成网格
    
    Args:
        images: 图像列表
        titles: 标题列表
        cell_size: 每个单元格大小
    
    Returns:
        拼接后的图像
    """
    n = len(images)
    if n == 0:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    
    # 计算网格布局
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # 创建大画布
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        # 缩放图像
        resized = resize_with_padding(img, cell_size)
        
        # 添加标题
        cv2.putText(resized, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 放置到网格中
        y1 = row * cell_size
        x1 = col * cell_size
        grid[y1:y1+cell_size, x1:x1+cell_size] = resized
    
    return grid


class MultiscaleVisualizer:
    """多尺度检测可视化类"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        初始化可视化器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.model_name = self.model_path.parent.parent.name  # 获取模型名称
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Warmup: 做一次空推理将模型移到GPU
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, device=device, verbose=False)
        LOGGER.info(f"✅ 模型加载成功 (已移至 {device})")
    
    def predict_single_scale(self, image: np.ndarray, scale: int, 
                            conf: float = 0.25, iou: float = 0.5) -> List:
        """
        单尺度推理
        
        Args:
            image: 输入图像
            scale: 推理尺度
            conf: 置信度阈值
            iou: NMS IoU阈值
        
        Returns:
            检测框列表 [(cls, x1, y1, x2, y2), ...]
        """
        results = self.model.predict(
            source=image,
            imgsz=scale,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
            save=False,
        )
        
        result = results[0]
        boxes = []
        
        if len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()  # 获取置信度
            
            for i in range(len(xyxy)):
                # 格式: (cls, conf, x1, y1, x2, y2)
                boxes.append((int(cls[i]), float(conf[i]), xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3]))
        
        return boxes
    
    def nms_fusion(self, all_boxes: List[List], iou_threshold: float = 0.5, 
                   class_agnostic: bool = True) -> List:
        """
        NMS融合多尺度结果
        
        Args:
            all_boxes: 所有尺度的检测框 [(cls, conf, x1, y1, x2, y2), ...]
            iou_threshold: IoU阈值
            class_agnostic: 是否使用跨类别NMS（默认True，解决多标签重复问题）
        
        Returns:
            融合后的检测框 [(cls, conf, x1, y1, x2, y2), ...]
        """
        # 合并所有框
        merged = []
        for boxes in all_boxes:
            merged.extend(boxes)
        
        if len(merged) == 0:
            return []
        
        LOGGER.info(f"   融合前: {len(merged)} 个检测框")
        
        # 使用跨类别NMS
        if class_agnostic:
            result = cross_class_nms(merged, iou_threshold)
            LOGGER.info(f"   跨类别NMS: {len(merged)} -> {len(result)}")
            return result
        else:
            # 按类别NMS（原始方法）
            boxes_array = np.array([[b[2], b[3], b[4], b[5]] for b in merged])  # x1,y1,x2,y2
            scores = np.array([b[1] for b in merged])  # 使用置信度作为score
            classes = np.array([b[0] for b in merged])
            
            keep_indices = []
            for cls in np.unique(classes):
                cls_mask = classes == cls
                cls_boxes = torch.from_numpy(boxes_array[cls_mask]).float()
                cls_scores = torch.from_numpy(scores[cls_mask]).float()
                
                keep = torch.ops.torchvision.nms(cls_boxes, cls_scores, iou_threshold)
                cls_indices = np.where(cls_mask)[0]
                keep_indices.extend(cls_indices[keep.numpy()].tolist())
            
            result = [merged[i] for i in keep_indices]
            LOGGER.info(f"   按类别NMS: {len(merged)} -> {len(result)}")
            return result
    
    def visualize_image(
        self,
        image_path: str,
        label_path: str,
        scales: List[int],
        save_dir: str,
        conf: float = 0.25,
        iou: float = 0.5,
        cell_size: int = 640,
        class_agnostic_nms: bool = True,
    ) -> str:
        """
        可视化单张图像的多尺度检测结果
        
        Args:
            image_path: 图像路径
            label_path: 标签路径
            scales: 尺度列表
            save_dir: 保存目录
            conf: 置信度阈值
            iou: NMS IoU阈值
            cell_size: 单元格大小
            class_agnostic_nms: 是否使用跨类别NMS（默认True，解决多标签重复问题）
        
        Returns:
            保存路径
        """
        image_path = Path(image_path)
        label_path = Path(label_path)
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        h, w = image.shape[:2]
        LOGGER.info(f"📸 处理图像: {image_path.name} ({w}x{h})")
        LOGGER.info(f"   图像路径: {image_path}")
        LOGGER.info(f"   标签路径: {label_path}")
        
        # 解析ground truth
        gt_boxes = parse_yolo_label(str(label_path), w, h)
        if len(gt_boxes) == 0:
            LOGGER.warning(f"   ⚠️ 未检测到Ground Truth标注！请检查标签文件")
        else:
            LOGGER.info(f"   ✅ Ground Truth: {len(gt_boxes)} 个目标")
        
        # 准备可视化图像列表
        vis_images = []
        vis_titles = []
        
        # 1. Ground Truth（按类别着色）
        gt_img = draw_boxes(image, gt_boxes, label="GT", show_class=True)
        vis_images.append(gt_img)
        vis_titles.append(f"Ground Truth ({len(gt_boxes)})")
        
        # 2. 每个尺度的检测结果（按类别着色）
        all_scale_boxes = []
        
        for i, scale in enumerate(scales):
            boxes = self.predict_single_scale(image, scale, conf, iou)
            all_scale_boxes.append(boxes)
            
            # 不指定颜色，让其按类别自动分配
            scale_img = draw_boxes(image, boxes, label=f"Scale {scale}", show_class=True)
            vis_images.append(scale_img)
            vis_titles.append(f"Scale {scale} ({len(boxes)})")
            
            LOGGER.info(f"   Scale {scale}: {len(boxes)} 个检测")
        
        # 3. 融合结果（按类别着色）
        fused_boxes = self.nms_fusion(all_scale_boxes, iou, class_agnostic=class_agnostic_nms)
        fused_img = draw_boxes(image, fused_boxes, label="Fused", show_class=True)
        vis_images.append(fused_img)
        vis_titles.append(f"Fused ({len(fused_boxes)})")
        
        LOGGER.info(f"   最终融合: {len(fused_boxes)} 个检测")
        
        # 创建网格图像
        grid = create_grid_image(vis_images, vis_titles, cell_size)
        
        # 保存
        scale_str = "_".join(map(str, scales))
        save_path = Path(save_dir) / f"{self.model_name}_{scale_str}" / f"{image_path.stem}_multiscale.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), grid)
        LOGGER.info(f"   ✅ 保存: {save_path}")
        
        return str(save_path)
    
    def visualize_directory(
        self,
        image_dir: str,
        label_dir: str,
        scales: List[int],
        save_dir: str,
        conf: float = 0.25,
        iou: float = 0.5,
        cell_size: int = 640,
        max_images: int = None,
        class_agnostic_nms: bool = True,
    ):
        """
        批量可视化目录中的图像
        
        Args:
            image_dir: 图像目录
            label_dir: 标签目录
            scales: 尺度列表
            save_dir: 保存目录
            conf: 置信度阈值
            iou: NMS IoU阈值
            cell_size: 单元格大小
            max_images: 最大处理图像数
            class_agnostic_nms: 是否使用跨类别NMS（默认True，解决多标签重复问题）
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        # 获取图像列表
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        LOGGER.info(f"🎯 开始批量可视化 ({len(image_files)} 张图像)")
        LOGGER.info(f"   图像目录: {image_dir}")
        LOGGER.info(f"   标签目录: {label_dir}")
        
        for i, img_path in enumerate(image_files, 1):
            label_path = label_dir / f"{img_path.stem}.txt"
            
            LOGGER.info(f"\n[{i}/{len(image_files)}]")
            try:
                self.visualize_image(
                    str(img_path), str(label_path), scales, save_dir, conf, iou, cell_size, class_agnostic_nms
                )
            except Exception as e:
                LOGGER.error(f"   ❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        LOGGER.info(f"\n✅ 批量可视化完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多尺度检测结果可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--image-dir", type=str, help="图像目录（批量处理）")
    parser.add_argument("--label", type=str, help="单张标签路径")
    parser.add_argument("--label-dir", type=str, help="标签目录（批量处理）")
    parser.add_argument("--scales", type=int, nargs="+", default=[640, 832, 1024, 1280],
                       help="尺度列表")
    parser.add_argument("--save-dir", type=str, default="runs/multiscale_visible", help="保存目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU阈值")
    parser.add_argument("--cell-size", type=int, default=640, help="单元格大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--max-images", type=int, default=None, help="最大处理图像数")
    parser.add_argument("--no-cross-class-nms", action="store_true", 
                       help="禁用跨类别NMS（默认启用，解决多标签重复问题）")
    
    args = parser.parse_args()
    
    try:
        visualizer = MultiscaleVisualizer(args.model, args.device)
        class_agnostic_nms = not args.no_cross_class_nms
        
        LOGGER.info(f"🔧 跨类别NMS: {'启用' if class_agnostic_nms else '禁用'}")
        
        if args.image and args.label:
            # 单张图像
            visualizer.visualize_image(
                args.image, args.label, args.scales, args.save_dir,
                args.conf, args.iou, args.cell_size, class_agnostic_nms
            )
        elif args.image_dir and args.label_dir:
            # 批量处理
            visualizer.visualize_directory(
                args.image_dir, args.label_dir, args.scales, args.save_dir,
                args.conf, args.iou, args.cell_size, args.max_images, class_agnostic_nms
            )
        else:
            LOGGER.error("❌ 请提供 --image/--label 或 --image-dir/--label-dir")
            
    except Exception as e:
        LOGGER.error(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

