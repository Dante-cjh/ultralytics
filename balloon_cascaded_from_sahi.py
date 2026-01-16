#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAHI切片推理结果的二阶段分类

功能：
1. 读取SAHI切片推理后的labels（YOLO格式）
2. 对每个检测框使用MobileNetV2分类器进行二次验证
3. 过滤误检，输出精修后的结果
4. 支持批量处理整个数据集
"""

import os
import sys
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 导入分类器模型
sys.path.insert(0, str(Path(__file__).parent))
from balloon_cascaded_detection import MobileNetClassifier, SimpleMLP


class SAHIResultRefiner:
    """SAHI结果精修器（使用二阶段分类器）"""
    
    def __init__(self, classifier_path: str, classifier_type: str = "mobilenet",
                 input_size: int = 112, num_classes: int = 2,
                 threshold: float = 0.5, device: str = "cuda:0"):
        """
        初始化精修器
        
        Args:
            classifier_path: 分类器权重路径
            classifier_type: 分类器类型 ('mlp' 或 'mobilenet')
            input_size: 分类器输入尺寸
            num_classes: 类别数（包括背景）
            threshold: 分类阈值（低于此值视为背景）
            device: 设备
        """
        self.device = device
        self.threshold = threshold
        self.input_size = input_size
        
        # 加载分类器
        if classifier_type == "mlp":
            self.classifier = SimpleMLP(input_size, num_classes)
        else:
            self.classifier = MobileNetClassifier(num_classes)
        
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 加载SAHI结果精修器")
        print(f"   分类器: {classifier_path}")
        print(f"   类型: {classifier_type}")
        print(f"   阈值: {threshold}")
    
    def parse_yolo_label(self, label_path: str, img_w: int, img_h: int) -> List[Dict]:
        """
        解析YOLO格式标签文件
        
        Returns:
            [{'cls': int, 'conf': float, 'box': [x1, y1, x2, y2]}, ...]
        """
        detections = []
        
        if not os.path.exists(label_path):
            return detections
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                    
                    # 如果有置信度
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    
                    detections.append({
                        'cls': cls,
                        'conf': conf,
                        'box': [x1, y1, x2, y2]
                    })
        
        return detections
    
    def refine_detections(self, image_path: str, label_path: str) -> List[Dict]:
        """
        精修SAHI检测结果
        
        Args:
            image_path: 原始图像路径
            label_path: SAHI推理结果label路径（YOLO格式）
        
        Returns:
            精修后的检测结果 [{'cls': int, 'conf': float, 'box': [x1,y1,x2,y2], 
                              'stage1_cls': int, 'stage1_conf': float}, ...]
        """
        # 读取图像
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        
        # 解析SAHI结果
        stage1_detections = self.parse_yolo_label(label_path, img_w, img_h)
        
        refined_detections = []
        
        for det in stage1_detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 裁剪候选区域
            crop = img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # 分类器推理
            crop_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classifier(crop_tensor)
                probs = F.softmax(output, dim=1)
                stage2_conf, stage2_cls = probs.max(1)
                stage2_conf = stage2_conf.item()
                stage2_cls = stage2_cls.item()
            
            # 过滤：如果分类器判断为背景（cls=0）或置信度低，则丢弃
            if stage2_cls == 0 or stage2_conf < self.threshold:
                continue
            
            # 转换类别（分类器的类别从1开始，需要减1）
            final_cls = stage2_cls - 1
            
            refined_detections.append({
                'cls': final_cls,
                'conf': stage2_conf,
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'stage1_cls': det['cls'],
                'stage1_conf': det['conf']
            })
        
        return refined_detections
    
    def save_yolo_label(self, detections: List[Dict], save_path: str, img_w: int, img_h: int):
        """保存为YOLO格式标签"""
        with open(save_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['box']
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                f.write(f"{det['cls']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    def visualize_comparison(self, image_path: str, stage1_dets: List[Dict], 
                            stage2_dets: List[Dict], save_path: str, class_names: List[str]):
        """可视化对比：SAHI结果 vs 二阶段精修结果"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # SAHI结果
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"SAHI结果 ({len(stage1_dets)} 检测)", fontsize=16)
        axes[0].axis('off')
        
        for det in stage1_dets:
            x1, y1, x2, y2 = det['box']
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                     edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            
            cls_name = class_names[det['cls']] if det['cls'] < len(class_names) else f"cls{det['cls']}"
            axes[0].text(x1, y1-5, f"{cls_name} {det['conf']:.2f}",
                        color='red', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 二阶段精修结果
        axes[1].imshow(img_rgb)
        axes[1].set_title(f"二阶段精修 ({len(stage2_dets)} 检测)", fontsize=16)
        axes[1].axis('off')
        
        for det in stage2_dets:
            x1, y1, x2, y2 = det['box']
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                     edgecolor='green', facecolor='none')
            axes[1].add_patch(rect)
            
            cls_name = class_names[det['cls']] if det['cls'] < len(class_names) else f"cls{det['cls']}"
            axes[1].text(x1, y1-5, f"{cls_name} {det['conf']:.2f}",
                        color='green', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def refine_dataset(sahi_results_dir: str, images_dir: str, classifier_path: str,
                   save_dir: str, classifier_type: str = "mobilenet",
                   input_size: int = 112, num_classes: int = 2,
                   threshold: float = 0.5, device: str = "cuda:0",
                   visualize: bool = True, class_names: List[str] = None):
    """
    批量精修SAHI结果
    
    Args:
        sahi_results_dir: SAHI推理结果目录（包含labels子目录）
        images_dir: 原始图像目录
        classifier_path: 分类器权重路径
        save_dir: 保存目录
        其他参数同SAHIResultRefiner
    """
    # 创建保存目录
    save_path = Path(save_dir)
    labels_save_dir = save_path / "labels"
    vis_save_dir = save_path / "visualizations"
    labels_save_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化精修器
    refiner = SAHIResultRefiner(
        classifier_path, classifier_type, input_size,
        num_classes, threshold, device
    )
    
    # 获取所有label文件
    sahi_labels_dir = Path(sahi_results_dir) / "labels"
    label_files = list(sahi_labels_dir.glob("*.txt"))
    
    print(f"\n📂 处理SAHI结果...")
    print(f"   SAHI结果: {sahi_results_dir}")
    print(f"   图像目录: {images_dir}")
    print(f"   保存目录: {save_dir}")
    print(f"   标签文件数: {len(label_files)}")
    
    # 统计信息
    total_stage1 = 0
    total_stage2 = 0
    filtered_count = 0
    
    # 处理每个文件
    for label_file in tqdm(label_files, desc="精修SAHI结果"):
        # 找到对应的图像
        img_name = label_file.stem
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = Path(images_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break
        
        if img_path is None:
            print(f"⚠️  找不到图像: {img_name}")
            continue
        
        # 精修检测结果
        stage1_dets = refiner.parse_yolo_label(str(label_file), 
                                               cv2.imread(img_path).shape[1],
                                               cv2.imread(img_path).shape[0])
        stage2_dets = refiner.refine_detections(img_path, str(label_file))
        
        # 统计
        total_stage1 += len(stage1_dets)
        total_stage2 += len(stage2_dets)
        filtered_count += (len(stage1_dets) - len(stage2_dets))
        
        # 保存精修后的labels
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        save_label_path = labels_save_dir / label_file.name
        refiner.save_yolo_label(stage2_dets, str(save_label_path), img_w, img_h)
        
        # 可视化
        if visualize:
            vis_path = vis_save_dir / f"{img_name}_comparison.jpg"
            if class_names is None:
                class_names = [f"cls{i}" for i in range(num_classes)]
            refiner.visualize_comparison(img_path, stage1_dets, stage2_dets,
                                        str(vis_path), class_names)
    
    # 打印统计信息
    print(f"\n✅ 精修完成!")
    print(f"   处理图像数: {len(label_files)}")
    print(f"   SAHI检测总数: {total_stage1}")
    print(f"   精修后检测总数: {total_stage2}")
    print(f"   过滤检测数: {filtered_count} ({100*filtered_count/max(total_stage1,1):.1f}%)")
    print(f"   保留率: {100*total_stage2/max(total_stage1,1):.1f}%")
    print(f"\n📁 结果保存至:")
    print(f"   Labels: {labels_save_dir}")
    if visualize:
        print(f"   可视化: {vis_save_dir}")


def main():
    parser = argparse.ArgumentParser(description="SAHI结果的二阶段精修")
    
    # 输入输出
    parser.add_argument('--sahi-results', type=str, required=True,
                       help='SAHI推理结果目录（包含labels子目录）')
    parser.add_argument('--images', type=str, required=True,
                       help='原始图像目录')
    parser.add_argument('--classifier', type=str, required=True,
                       help='分类器权重路径')
    parser.add_argument('--save-dir', type=str, required=True,
                       help='保存目录')
    
    # 分类器参数
    parser.add_argument('--model-type', type=str, default='mobilenet',
                       choices=['mlp', 'mobilenet'], help='分类器类型')
    parser.add_argument('--input-size', type=int, default=112,
                       help='分类器输入尺寸')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='类别数（包括背景）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='分类阈值')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--no-visualize', action='store_true',
                       help='不生成可视化图像')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='类别名称列表')
    
    args = parser.parse_args()
    
    # 运行精修
    refine_dataset(
        sahi_results_dir=args.sahi_results,
        images_dir=args.images,
        classifier_path=args.classifier,
        save_dir=args.save_dir,
        classifier_type=args.model_type,
        input_size=args.input_size,
        num_classes=args.num_classes,
        threshold=args.threshold,
        device=args.device,
        visualize=not args.no_visualize,
        class_names=args.class_names
    )


if __name__ == '__main__':
    main()

