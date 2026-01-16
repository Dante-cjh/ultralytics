#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多置信度检测结果可视化脚本
将ground truth、每个置信度的检测结果拼接到一张图上

使用方法:
python visualize_multi_conf_results.py \
    --model best.pt \
    --image /path/to/image.jpg \
    --label /path/to/label.txt \
    --conf-list 0.1 0.2 0.3 0.4 0.5 \
    --save-dir runs/multiconf_visible
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import math

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """解析YOLO格式标签文件"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append((cls, x1, y1, x2, y2))
    
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
        class_names: 类别名称字典
        show_conf: 是否显示置信度
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    # 类别颜色映射
    class_colors = {
        0: (0, 255, 0),      # 绿色 - class 0
        1: (255, 0, 0),      # 蓝色 - class 1
        2: (0, 0, 255),      # 红色 - class 2
        3: (255, 255, 0),    # 青色 - class 3
        4: (255, 0, 255),    # 紫色 - class 4
        5: (0, 255, 255),    # 黄色 - class 5
    }
    
    if class_names is None:
        class_names = {0: 'hole', 1: 'cave', 2: 'unknow'}
    
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
            
            box_color = color if color is not None else class_colors.get(cls, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
            
            # 绘制类别和置信度标签
            if show_class or (show_conf and conf is not None):
                class_name = class_names.get(cls, f'cls{cls}')
                
                # 组合标签文字
                if conf is not None and show_conf:
                    label_text = f"{class_name} {conf:.2f}"
                else:
                    label_text = f"{class_name}"
                
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_y = max(y1 - 5, text_h + 5)
                cv2.rectangle(img, (x1, label_y - text_h - baseline),
                            (x1 + text_w, label_y), box_color, -1)
                cv2.putText(img, label_text, (x1, label_y - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 添加图片标题
    if label:
        cv2.putText(img, f"{label} ({len(boxes)})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img


def resize_with_padding(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """等比例缩放图像并填充到目标尺寸"""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def create_grid_image(images: List[np.ndarray], titles: List[str], 
                      cell_size: int = 640) -> np.ndarray:
    """将多张图像拼接成网格"""
    n = len(images)
    if n == 0:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        resized = resize_with_padding(img, cell_size)
        
        cv2.putText(resized, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y1 = row * cell_size
        x1 = col * cell_size
        grid[y1:y1+cell_size, x1:x1+cell_size] = resized
    
    return grid


class MultiConfVisualizer:
    """多置信度检测可视化类"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """初始化可视化器"""
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.model_name = self.model_path.parent.parent.name
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Warmup: 做一次空推理将模型移到GPU
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, device=device, verbose=False)
        LOGGER.info(f"✅ 模型加载成功 (已移至 {device})")
    
    def predict_with_conf(self, image: np.ndarray, imgsz: int, 
                         conf: float, iou: float = 0.5) -> List:
        """使用指定置信度推理，返回带置信度的检测框"""
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
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
            conf_scores = result.boxes.conf.cpu().numpy()  # 获取置信度
            
            for i in range(len(xyxy)):
                # 格式: (cls, conf, x1, y1, x2, y2)
                boxes.append((int(cls[i]), float(conf_scores[i]), xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3]))
        
        return boxes
    
    def visualize_image(
        self,
        image_path: str,
        label_path: str,
        conf_list: List[float],
        save_dir: str,
        imgsz: int = 1280,
        iou: float = 0.5,
        cell_size: int = 640,
    ) -> str:
        """可视化单张图像的多置信度检测结果"""
        image_path = Path(image_path)
        label_path = Path(label_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        h, w = image.shape[:2]
        LOGGER.info(f"📸 处理图像: {image_path.name} ({w}x{h})")
        
        gt_boxes = parse_yolo_label(str(label_path), w, h)
        LOGGER.info(f"   Ground Truth: {len(gt_boxes)} 个目标")
        
        vis_images = []
        vis_titles = []
        
        # 1. Ground Truth（按类别着色）
        gt_img = draw_boxes(image, gt_boxes, label="GT", show_class=True)
        vis_images.append(gt_img)
        vis_titles.append(f"GT ({len(gt_boxes)})")
        
        # 2. 每个置信度的检测结果
        colors = [
            (255, 0, 0),    # 蓝
            (0, 165, 255),  # 橙
            (255, 255, 0),  # 青
            (147, 20, 255), # 粉
            (0, 255, 255),  # 黄
            (0, 0, 255),    # 红
            (255, 0, 255),  # 紫
            (128, 128, 0),  # 深青
        ]
        
        for i, conf in enumerate(conf_list):
            boxes = self.predict_with_conf(image, imgsz, conf, iou)
            
            # 不指定颜色，让其按类别自动分配
            conf_img = draw_boxes(image, boxes, label=f"conf={conf}", show_class=True)
            vis_images.append(conf_img)
            vis_titles.append(f"conf={conf} ({len(boxes)})")
            
            LOGGER.info(f"   conf={conf}: {len(boxes)} 个检测")
        
        # 创建网格图像
        grid = create_grid_image(vis_images, vis_titles, cell_size)
        
        # 保存
        conf_str = "_".join([f"{c:.2f}" for c in conf_list])
        save_path = Path(save_dir) / f"{self.model_name}_conf_{conf_str}" / f"{image_path.stem}_multiconf.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), grid)
        LOGGER.info(f"   ✅ 保存: {save_path}")
        
        return str(save_path)
    
    def visualize_directory(
        self,
        image_dir: str,
        label_dir: str,
        conf_list: List[float],
        save_dir: str,
        imgsz: int = 1280,
        iou: float = 0.5,
        cell_size: int = 640,
        max_images: int = None,
    ):
        """批量可视化目录中的图像"""
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        LOGGER.info(f"🎯 开始批量可视化 ({len(image_files)} 张图像)")
        
        for i, img_path in enumerate(image_files, 1):
            label_path = label_dir / f"{img_path.stem}.txt"
            
            LOGGER.info(f"\n[{i}/{len(image_files)}]")
            try:
                self.visualize_image(
                    str(img_path), str(label_path), conf_list, save_dir,
                    imgsz, iou, cell_size
                )
            except Exception as e:
                LOGGER.error(f"   ❌ 处理失败: {e}")
        
        LOGGER.info(f"\n✅ 批量可视化完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多置信度检测结果可视化",
    )
    
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--image-dir", type=str, help="图像目录（批量处理）")
    parser.add_argument("--label", type=str, help="单张标签路径")
    parser.add_argument("--label-dir", type=str, help="标签目录（批量处理）")
    parser.add_argument("--conf-list", type=float, nargs="+", 
                       default=[0.1, 0.2, 0.3, 0.4, 0.5],
                       help="置信度列表")
    parser.add_argument("--save-dir", type=str, default="runs/multiconf_visible", help="保存目录")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU阈值")
    parser.add_argument("--cell-size", type=int, default=640, help="单元格大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--max-images", type=int, default=None, help="最大处理图像数")
    
    args = parser.parse_args()
    
    try:
        visualizer = MultiConfVisualizer(args.model, args.device)
        
        if args.image and args.label:
            visualizer.visualize_image(
                args.image, args.label, args.conf_list, args.save_dir,
                args.imgsz, args.iou, args.cell_size
            )
        elif args.image_dir and args.label_dir:
            visualizer.visualize_directory(
                args.image_dir, args.label_dir, args.conf_list, args.save_dir,
                args.imgsz, args.iou, args.cell_size, args.max_images
            )
        else:
            LOGGER.error("❌ 请提供 --image/--label 或 --image-dir/--label-dir")
            
    except Exception as e:
        LOGGER.error(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

