#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SAHI的两阶段级联检测 - 批量推理

功能：
1. 使用SAHI对整个数据集进行切片推理
2. 对SAHI的每个检测框使用MobileNetV2进行二次分类
3. 保存精修后的结果（images + labels + 可视化）
4. 生成评估报告
"""

import os
import sys
import cv2
import yaml
import json
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
from collections import defaultdict

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 导入分类器模型
sys.path.insert(0, str(Path(__file__).parent))
from balloon_cascaded_detection import MobileNetClassifier, SimpleMLP, parse_yolo_label


class SAHICascadedDetector:
    """基于SAHI的两阶段级联检测器"""
    
    def __init__(self, yolo_model_path: str, classifier_path: str,
                 classifier_type: str = "mobilenet", input_size: int = 112,
                 num_classes: int = 2, device: str = "cuda:0",
                 # SAHI参数
                 slice_height: int = 640, slice_width: int = 640,
                 overlap_ratio: float = 0.2, conf_threshold: float = 0.25,
                 # 二阶段参数
                 stage2_threshold: float = 0.5):
        """
        初始化SAHI两阶段检测器
        
        Args:
            yolo_model_path: YOLO模型路径
            classifier_path: 分类器权重路径
            classifier_type: 分类器类型 ('mlp' 或 'mobilenet')
            input_size: 分类器输入尺寸
            num_classes: 类别数（包括背景）
            device: 设备
            slice_height: SAHI切片高度
            slice_width: SAHI切片宽度
            overlap_ratio: SAHI重叠比例
            conf_threshold: SAHI置信度阈值
            stage2_threshold: 二阶段分类阈值
        """
        self.device = device
        self.stage2_threshold = stage2_threshold
        self.input_size = input_size
        
        # 加载SAHI检测模型
        print(f"✅ 加载SAHI检测模型...")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_model_path,
            confidence_threshold=conf_threshold,
            device=device
        )
        
        # SAHI参数
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        
        print(f"   YOLO模型: {yolo_model_path}")
        print(f"   切片尺寸: {slice_height}x{slice_width}")
        print(f"   重叠比例: {overlap_ratio}")
        print(f"   置信度阈值: {conf_threshold}")
        
        # 加载二阶段分类器
        print(f"✅ 加载二阶段分类器...")
        if classifier_type == "mlp":
            self.classifier = SimpleMLP(input_size, num_classes)
        else:
            self.classifier = MobileNetClassifier(num_classes)
        
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        
        print(f"   分类器: {classifier_path}")
        print(f"   类型: {classifier_type}")
        print(f"   阈值: {stage2_threshold}")
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_with_sahi(self, image_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        使用SAHI进行切片推理，然后使用二阶段分类器精修
        
        Args:
            image_path: 图像路径
        
        Returns:
            (stage1_detections, stage2_detections)
            stage1_detections: SAHI原始检测结果
            stage2_detections: 二阶段精修后的结果
        """
        # 读取图像
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        
        # SAHI切片推理
        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5
        )
        
        # 提取SAHI检测结果
        stage1_detections = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            stage1_detections.append({
                'box': [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                'cls': obj.category.id,
                'conf': obj.score.value
            })
        
        # 二阶段分类
        stage2_detections = []
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
            if stage2_cls == 0 or stage2_conf < self.stage2_threshold:
                continue
            
            # 转换类别（分类器的类别从1开始，需要减1）
            final_cls = stage2_cls - 1
            
            stage2_detections.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'cls': final_cls,
                'conf': stage2_conf,
                'stage1_cls': det['cls'],
                'stage1_conf': det['conf']
            })
        
        return stage1_detections, stage2_detections
    
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
    
    def draw_detections(self, img: np.ndarray, detections: List[Dict], 
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """在图像上绘制检测框"""
        img_draw = img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            label = f"cls{det['cls']} {det['conf']:.2f}"
            cv2.putText(img_draw, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img_draw
    
    def visualize_comparison(self, img: np.ndarray, stage1_dets: List[Dict],
                            stage2_dets: List[Dict], save_path: str):
        """可视化SAHI vs 二阶段对比"""
        img_h, img_w = img.shape[:2]
        
        # 绘制SAHI结果
        img_sahi = self.draw_detections(img, stage1_dets, color=(0, 0, 255))  # 红色
        
        # 绘制二阶段结果
        img_stage2 = self.draw_detections(img, stage2_dets, color=(0, 255, 0))  # 绿色
        
        # 拼接两张图像
        gap = np.ones((img_h, 20, 3), dtype=np.uint8) * 255
        vis_img = np.hstack([img_sahi, gap, img_stage2])
        
        # 添加标题
        title_height = 50
        title_bar = np.ones((title_height, vis_img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_bar, f"SAHI ({len(stage1_dets)} dets)", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(title_bar, f"SAHI + Stage2 ({len(stage2_dets)} dets)", 
                   (img_w + 30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        vis_img = np.vstack([title_bar, vis_img])
        
        # 保存
        cv2.imwrite(save_path, vis_img)
    
    def count_objects(self, detections: List[Dict]) -> Dict[int, int]:
        """统计每个类别的目标数量"""
        counts = defaultdict(int)
        for det in detections:
            counts[det['cls']] += 1
        return dict(counts)
    
    def count_gt_objects(self, label_path: str, img_w: int, img_h: int) -> Dict[int, int]:
        """统计GT中每个类别的目标数量"""
        gt_boxes = parse_yolo_label(label_path, img_w, img_h)
        counts = defaultdict(int)
        for box in gt_boxes:
            counts[int(box[0])] += 1
        return dict(counts)
    
    def calculate_count_accuracy(self, pred_counts: Dict[int, int], 
                                 gt_counts: Dict[int, int]) -> float:
        """计算数量准确率: 1 - |predict-true|/true"""
        all_classes = set(list(pred_counts.keys()) + list(gt_counts.keys()))
        
        total_error = 0
        total_gt = 0
        
        for cls in all_classes:
            pred = pred_counts.get(cls, 0)
            gt = gt_counts.get(cls, 0)
            
            if gt > 0:
                error = abs(pred - gt) / gt
                total_error += error * gt
                total_gt += gt
        
        if total_gt == 0:
            return 1.0
        
        accuracy = 1.0 - (total_error / total_gt)
        return max(0.0, accuracy)


def evaluate_dataset(yolo_model: str, classifier: str, data_yaml: str,
                     split: str = "val", save_dir: str = "runs/sahi_cascaded_eval",
                     classifier_type: str = "mobilenet", input_size: int = 112,
                     num_classes: int = 2, device: str = "cuda:0",
                     # SAHI参数
                     slice_height: int = 640, slice_width: int = 640,
                     overlap_ratio: float = 0.2, sahi_conf: float = 0.25,
                     # 二阶段参数
                     stage2_threshold: float = 0.5):
    """
    在整个数据集上进行SAHI两阶段批量推理和评估
    
    Args:
        yolo_model: YOLO模型路径
        classifier: 分类器权重路径
        data_yaml: 数据集YAML配置
        split: 'train' 或 'val'
        save_dir: 保存目录
        其他参数见SAHICascadedDetector
    """
    # 读取数据集配置
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config['path'])
    # 支持两种目录结构
    if (dataset_path / 'images' / split).exists():
        image_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
    else:
        image_dir = dataset_path / data_config[split] / 'images'
        label_dir = dataset_path / data_config[split] / 'labels'
    
    # 创建保存目录
    save_path = Path(save_dir)
    images_dir = save_path / 'images'
    labels_dir_sahi = save_path / 'labels_sahi'
    labels_dir_stage2 = save_path / 'labels_sahi_stage2'
    vis_comp_dir = save_path / 'visualizations_comparison'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir_sahi.mkdir(exist_ok=True)
    labels_dir_stage2.mkdir(exist_ok=True)
    vis_comp_dir.mkdir(exist_ok=True)
    
    # 初始化检测器
    detector = SAHICascadedDetector(
        yolo_model, classifier, classifier_type, input_size,
        num_classes, device, slice_height, slice_width,
        overlap_ratio, sahi_conf, stage2_threshold
    )
    
    # 获取所有图像
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    
    print(f"\n🔍 SAHI两阶段批量推理 ({len(image_files)} 张图像)...")
    print(f"   数据集: {data_yaml}")
    print(f"   划分: {split}")
    print(f"   保存目录: {save_dir}")
    
    # 结果存储
    results = {
        'sahi': [],
        'sahi_stage2': [],
        'comparison': []
    }
    
    # 统计信息
    stats = {
        'sahi': {
            'count_accuracies': [],
            'total_detections': 0,
            'total_gt': 0
        },
        'sahi_stage2': {
            'count_accuracies': [],
            'total_detections': 0,
            'total_gt': 0
        }
    }
    
    # 处理每张图像
    for img_path in tqdm(image_files, desc="推理中"):
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        
        # 读取GT
        label_path = label_dir / (img_path.stem + '.txt')
        gt_counts = detector.count_gt_objects(str(label_path), img_w, img_h)
        total_gt = sum(gt_counts.values())
        
        # SAHI两阶段检测
        stage1_dets, stage2_dets = detector.detect_with_sahi(str(img_path))
        
        # 计算准确率
        sahi_counts = detector.count_objects(stage1_dets)
        sahi_acc = detector.calculate_count_accuracy(sahi_counts, gt_counts)
        
        stage2_counts = detector.count_objects(stage2_dets)
        stage2_acc = detector.calculate_count_accuracy(stage2_counts, gt_counts)
        
        # 记录结果
        results['sahi'].append({
            'image': img_path.name,
            'gt_counts': gt_counts,
            'pred_counts': sahi_counts,
            'count_accuracy': sahi_acc,
            'num_detections': len(stage1_dets)
        })
        
        results['sahi_stage2'].append({
            'image': img_path.name,
            'gt_counts': gt_counts,
            'pred_counts': stage2_counts,
            'count_accuracy': stage2_acc,
            'num_detections': len(stage2_dets)
        })
        
        results['comparison'].append({
            'image': img_path.name,
            'gt_total': total_gt,
            'sahi': {
                'count': sum(sahi_counts.values()),
                'accuracy': sahi_acc
            },
            'sahi_stage2': {
                'count': sum(stage2_counts.values()),
                'accuracy': stage2_acc
            },
            'improvement': stage2_acc - sahi_acc
        })
        
        # 更新统计
        stats['sahi']['count_accuracies'].append(sahi_acc)
        stats['sahi']['total_detections'] += len(stage1_dets)
        stats['sahi']['total_gt'] += total_gt
        
        stats['sahi_stage2']['count_accuracies'].append(stage2_acc)
        stats['sahi_stage2']['total_detections'] += len(stage2_dets)
        stats['sahi_stage2']['total_gt'] += total_gt
        
        # 保存推理图像（二阶段结果）
        img_stage2 = detector.draw_detections(img, stage2_dets)
        cv2.imwrite(str(images_dir / img_path.name), img_stage2)
        
        # 保存labels
        detector.save_yolo_label(stage1_dets, str(labels_dir_sahi / f"{img_path.stem}.txt"), img_w, img_h)
        detector.save_yolo_label(stage2_dets, str(labels_dir_stage2 / f"{img_path.stem}.txt"), img_w, img_h)
        
        # 保存可视化对比
        detector.visualize_comparison(img, stage1_dets, stage2_dets,
                                      str(vis_comp_dir / f"{img_path.stem}_comparison.jpg"))
    
    # 计算平均指标
    sahi_avg_acc = np.mean(stats['sahi']['count_accuracies']) * 100
    stage2_avg_acc = np.mean(stats['sahi_stage2']['count_accuracies']) * 100
    improvement = stage2_avg_acc - sahi_avg_acc
    
    # 保存详细结果
    with open(save_path / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = f"""
{'='*60}
SAHI两阶段级联检测 - 评估报告
{'='*60}

【总体指标】

SAHI切片推理:
  - 平均数量准确率: {sahi_avg_acc:.2f}%
  - 总检测数: {stats['sahi']['total_detections']}
  - 总GT数: {stats['sahi']['total_gt']}

SAHI + 二阶段分类:
  - 平均数量准确率: {stage2_avg_acc:.2f}%
  - 总检测数: {stats['sahi_stage2']['total_detections']}
  - 总GT数: {stats['sahi_stage2']['total_gt']}

性能提升:
  - 准确率提升: {improvement:+.2f}%
  - {'✅ 二阶段更优' if improvement > 0 else '⚠️ 二阶段未提升'}

{'='*60}

【详细分析】

提升最大的前5张图像:
"""
    
    # 排序找出提升最大的图像
    sorted_results = sorted(results['comparison'], key=lambda x: x['improvement'], reverse=True)
    for i, item in enumerate(sorted_results[:5], 1):
        report += f"  {i}. {item['image']}: {item['improvement']*100:+.2f}%\n"
    
    report += "\n下降最大的5张图像:\n"
    for i, item in enumerate(sorted_results[-5:], 1):
        report += f"  {i}. {item['image']}: {item['improvement']*100:+.2f}%\n"
    
    report += f"\n{'='*60}\n"
    
    # 保存和打印报告
    with open(save_path / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ 评估完成! 结果保存至: {save_path}")
    print(f"   推理图像: {images_dir}")
    print(f"   SAHI标签: {labels_dir_sahi}")
    print(f"   二阶段标签: {labels_dir_stage2}")
    print(f"   可视化对比: {vis_comp_dir}")


def main():
    parser = argparse.ArgumentParser(description="基于SAHI的两阶段级联检测 - 批量推理")
    
    # 模型参数
    parser.add_argument('--yolo-model', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--classifier', type=str, required=True, help='分类器权重路径')
    parser.add_argument('--data-yaml', type=str, required=True, help='数据集YAML配置')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], 
                       help='数据集划分')
    
    # 分类器参数
    parser.add_argument('--model-type', type=str, default='mobilenet',
                       choices=['mlp', 'mobilenet'], help='分类器类型')
    parser.add_argument('--input-size', type=int, default=112, help='分类器输入尺寸')
    parser.add_argument('--num-classes', type=int, default=2, help='类别数（包括背景）')
    
    # SAHI参数
    parser.add_argument('--slice-height', type=int, default=640, help='SAHI切片高度')
    parser.add_argument('--slice-width', type=int, default=640, help='SAHI切片宽度')
    parser.add_argument('--overlap-ratio', type=float, default=0.2, help='SAHI重叠比例')
    parser.add_argument('--sahi-conf', type=float, default=0.25, help='SAHI置信度阈值')
    
    # 二阶段参数
    parser.add_argument('--stage2-threshold', type=float, default=0.5, help='二阶段分类阈值')
    
    # 其他参数
    parser.add_argument('--save-dir', type=str, default='runs/sahi_cascaded_eval',
                       help='保存目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    # 运行评估
    evaluate_dataset(
        yolo_model=args.yolo_model,
        classifier=args.classifier,
        data_yaml=args.data_yaml,
        split=args.split,
        save_dir=args.save_dir,
        classifier_type=args.model_type,
        input_size=args.input_size,
        num_classes=args.num_classes,
        device=args.device,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_ratio=args.overlap_ratio,
        sahi_conf=args.sahi_conf,
        stage2_threshold=args.stage2_threshold
    )


if __name__ == '__main__':
    main()

