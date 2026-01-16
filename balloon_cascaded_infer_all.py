#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段级联检测 - 批量推理和评估脚本

功能：
1. 在整个验证集上进行两阶段推理
2. 计算数量准确率指标 (1 - |predict-true|/true)
3. 与单阶段YOLO对比
4. 生成详细的分析报告
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
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

# 导入分类器模型
sys.path.insert(0, str(Path(__file__).parent))
from balloon_cascaded_detection import MobileNetClassifier, SimpleMLP, parse_yolo_label, cross_class_nms


class CascadedEvaluator:
    """两阶段级联检测评估器"""
    
    def __init__(self, yolo_model_path: str, classifier_path: str,
                 classifier_type: str = "mobilenet", input_size: int = 112,
                 num_classes: int = 2, device: str = "cuda:0",
                 cross_class_nms: bool = True, nms_iou: float = 0.3):
        """
        初始化评估器
        
        Args:
            yolo_model_path: YOLO模型路径
            classifier_path: 分类器权重路径
            classifier_type: 分类器类型
            input_size: 分类器输入尺寸
            num_classes: 类别数（包括背景）
            device: 设备
            cross_class_nms: 是否使用跨类别NMS
            nms_iou: 跨类别NMS的IOU阈值
        """
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device)
        
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
        
        self.device = device
        self.use_cross_class_nms = cross_class_nms
        self.nms_iou = nms_iou
        
        print(f"✅ 加载两阶段检测器")
        print(f"   YOLO: {yolo_model_path}")
        print(f"   分类器: {classifier_path}")
        print(f"   跨类别NMS: {'启用' if cross_class_nms else '禁用'} (IOU={nms_iou})")
    
    def detect_single_stage(self, image_path: str, conf: float = 0.25, 
                           imgsz: int = 1280) -> List[Dict]:
        """
        单阶段YOLO检测（用于对比）
        
        Returns:
            [{'box': [x1,y1,x2,y2], 'cls': int, 'conf': float}, ...]
        """
        results = self.yolo_model.predict(
            source=image_path,
            imgsz=imgsz,
            conf=conf,
            iou=0.45,
            device=self.device,
            verbose=False,
            save=False,
        )
        
        result = results[0]
        detections = []
        
        if len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf_scores = result.boxes.conf.cpu().numpy()
            
            for i in range(len(xyxy)):
                detections.append({
                    'box': xyxy[i].tolist(),
                    'cls': int(cls[i]),
                    'conf': float(conf_scores[i])
                })
        
        return detections
    
    def detect_two_stage(self, image_path: str, stage1_conf: float = 0.05,
                        stage2_threshold: float = 0.5, imgsz: int = 1280) -> List[Dict]:
        """
        两阶段级联检测
        
        Returns:
            [{'box': [x1,y1,x2,y2], 'cls': int, 'conf': float, 'stage1_cls': int, 'stage1_conf': float}, ...]
        """
        # 读取图像
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        
        # 第一阶段：YOLO生成候选框
        results = self.yolo_model.predict(
            source=image_path,
            imgsz=imgsz,
            conf=stage1_conf,
            iou=0.45,
            device=self.device,
            verbose=False,
            save=False,
        )
        
        result = results[0]
        detections = []
        
        if len(result.boxes) == 0:
            return detections
        
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        
        # 第二阶段：分类器重分类
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
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
            if stage2_cls == 0 or stage2_conf < stage2_threshold:
                continue
            
            # 转换类别（分类器的类别从1开始，需要减1）
            final_cls = stage2_cls - 1
            
            detections.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'cls': final_cls,
                'conf': stage2_conf,
                'stage1_cls': int(cls[i]),
                'stage1_conf': float(conf[i])
            })
        
        # 跨类别NMS（可选）
        if self.use_cross_class_nms and len(detections) > 0:
            detections = cross_class_nms(detections, self.nms_iou)
        
        return detections
    
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
        """
        计算数量准确率: 1 - |predict-true|/true
        
        Args:
            pred_counts: 预测的每类数量 {cls: count}
            gt_counts: GT的每类数量 {cls: count}
        
        Returns:
            数量准确率 (0-1)
        """
        # 获取所有类别
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
        return max(0.0, accuracy)  # 确保不小于0
    
    def evaluate_dataset(self, data_yaml: str, split: str = "val",
                        stage1_conf: float = 0.05, stage2_threshold: float = 0.5,
                        yolo_conf: float = 0.25, imgsz: int = 1280,
                        save_dir: str = "runs/cascaded_eval"):
        """
        在整个数据集上评估
        
        Args:
            data_yaml: 数据集YAML配置
            split: 'train' 或 'val'
            stage1_conf: 第一阶段置信度
            stage2_threshold: 第二阶段阈值
            yolo_conf: 单阶段YOLO置信度（用于对比）
            imgsz: 推理尺寸
            save_dir: 保存目录
        """
        # 读取数据集配置
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_config['path'])
        # 支持两种目录结构：
        # 1. path/train/images 和 path/train/labels
        # 2. path/images/train 和 path/labels/train (balloon格式)
        if (dataset_path / 'images' / split).exists():
            # Balloon格式
            image_dir = dataset_path / 'images' / split
            label_dir = dataset_path / 'labels' / split
        else:
            # 标准格式
            image_dir = dataset_path / data_config[split] / 'images'
            label_dir = dataset_path / data_config[split] / 'labels'
        
        # 创建保存目录（类似runs/inference/<model_name>_val的结构）
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 创建images和labels目录
        images_dir = save_path / 'images'
        images_dir.mkdir(exist_ok=True)
        labels_dir_single = save_path / 'labels_single_stage'
        labels_dir_two = save_path / 'labels_two_stage'
        labels_dir_single.mkdir(exist_ok=True)
        labels_dir_two.mkdir(exist_ok=True)
        
        # 额外创建可视化对比目录
        vis_comp_dir = save_path / 'visualizations_comparison'
        vis_comp_dir.mkdir(exist_ok=True)
        
        # 结果存储
        results = {
            'single_stage': [],
            'two_stage': [],
            'comparison': []
        }
        
        # 统计信息
        stats = {
            'single_stage': {
                'count_accuracies': [],
                'total_detections': 0,
                'total_gt': 0
            },
            'two_stage': {
                'count_accuracies': [],
                'total_detections': 0,
                'total_gt': 0
            }
        }
        
        # 处理每张图像
        image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        
        print(f"\n🔍 评估 {split} 集 ({len(image_files)} 张图像)...")
        print(f"   单阶段YOLO置信度: {yolo_conf}")
        print(f"   两阶段配置: stage1_conf={stage1_conf}, stage2_threshold={stage2_threshold}")
        
        for img_path in tqdm(image_files, desc="评估中"):
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 读取GT
            label_path = label_dir / (img_path.stem + '.txt')
            gt_counts = self.count_gt_objects(str(label_path), img_w, img_h)
            total_gt = sum(gt_counts.values())
            
            # 单阶段检测
            single_dets = self.detect_single_stage(str(img_path), yolo_conf, imgsz)
            single_counts = self.count_objects(single_dets)
            single_acc = self.calculate_count_accuracy(single_counts, gt_counts)
            
            # 两阶段检测
            two_dets = self.detect_two_stage(str(img_path), stage1_conf, stage2_threshold, imgsz)
            two_counts = self.count_objects(two_dets)
            two_acc = self.calculate_count_accuracy(two_counts, gt_counts)
            
            # 记录结果
            results['single_stage'].append({
                'image': img_path.name,
                'gt_counts': gt_counts,
                'pred_counts': single_counts,
                'count_accuracy': single_acc,
                'num_detections': len(single_dets)
            })
            
            results['two_stage'].append({
                'image': img_path.name,
                'gt_counts': gt_counts,
                'pred_counts': two_counts,
                'count_accuracy': two_acc,
                'num_detections': len(two_dets)
            })
            
            results['comparison'].append({
                'image': img_path.name,
                'gt_total': total_gt,
                'single_stage': {
                    'count': sum(single_counts.values()),
                    'accuracy': single_acc
                },
                'two_stage': {
                    'count': sum(two_counts.values()),
                    'accuracy': two_acc
                },
                'improvement': two_acc - single_acc
            })
            
            # 更新统计
            stats['single_stage']['count_accuracies'].append(single_acc)
            stats['single_stage']['total_detections'] += len(single_dets)
            stats['single_stage']['total_gt'] += total_gt
            
            stats['two_stage']['count_accuracies'].append(two_acc)
            stats['two_stage']['total_detections'] += len(two_dets)
            stats['two_stage']['total_gt'] += total_gt
            
            # 保存推理图像、可视化和标签
            self._save_visualizations_and_labels(
                img, img_path.stem, single_dets, two_dets,
                images_dir, vis_comp_dir, labels_dir_single, labels_dir_two
            )
        
        # 计算平均指标
        single_avg_acc = np.mean(stats['single_stage']['count_accuracies']) * 100
        two_avg_acc = np.mean(stats['two_stage']['count_accuracies']) * 100
        improvement = two_avg_acc - single_avg_acc
        
        # 保存详细结果
        with open(save_path / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成报告
        report = self.generate_report(results, stats, single_avg_acc, two_avg_acc)
        
        with open(save_path / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n✅ 评估完成! 结果保存至: {save_path}")
    
    def _save_visualizations_and_labels(self, img: np.ndarray, img_name: str,
                                        single_dets: List[Dict], two_dets: List[Dict],
                                        images_dir: Path, vis_comp_dir: Path,
                                        labels_dir_single: Path, labels_dir_two: Path):
        """
        保存推理图像、可视化对比和YOLO格式标签
        
        Args:
            img: 原始图像
            img_name: 图像名称（不含扩展名）
            single_dets: 单阶段检测结果
            two_dets: 两阶段检测结果
            images_dir: 推理图像保存目录
            vis_comp_dir: 可视化对比保存目录
            labels_dir_single: 单阶段标签保存目录
            labels_dir_two: 两阶段标签保存目录
        """
        img_h, img_w = img.shape[:2]
        
        # 绘制两阶段推理图像（主要结果，保存到images目录）
        img_two_stage = img.copy()
        for det in two_dets:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cv2.rectangle(img_two_stage, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls{det['cls']} {det['conf']:.2f}"
            cv2.putText(img_two_stage, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存两阶段推理图像到images目录（主要结果）
        cv2.imwrite(str(images_dir / f"{img_name}.jpg"), img_two_stage)
        
        # 绘制对比可视化（左：单阶段，右：两阶段）
        img_single = img.copy()
        img_two = img.copy()
        
        # 绘制单阶段检测结果
        for det in single_dets:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cv2.rectangle(img_single, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色
            label = f"cls{det['cls']} {det['conf']:.2f}"
            cv2.putText(img_single, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制两阶段检测结果（绿色）
        for det in two_dets:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cv2.rectangle(img_two, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色
            label = f"cls{det['cls']} {det['conf']:.2f}"
            cv2.putText(img_two, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 拼接两张图像（对比可视化）
        gap = np.ones((img_h, 20, 3), dtype=np.uint8) * 255
        vis_img = np.hstack([img_single, gap, img_two])
        
        # 添加标题
        title_height = 50
        title_bar = np.ones((title_height, vis_img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_bar, f"Single-Stage ({len(single_dets)} dets)", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 红色
        cv2.putText(title_bar, f"Two-Stage ({len(two_dets)} dets)", 
                   (img_w + 30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # 绿色
        
        vis_img = np.vstack([title_bar, vis_img])
        
        # 保存对比可视化图像到visualizations_comparison目录
        cv2.imwrite(str(vis_comp_dir / f"{img_name}_comparison.jpg"), vis_img)
        
        # 保存YOLO格式标签
        # 单阶段
        with open(labels_dir_single / f"{img_name}.txt", 'w') as f:
            for det in single_dets:
                x1, y1, x2, y2 = det['box']
                cls = det['cls']
                # 转换为YOLO格式 (class x_center y_center width height)
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 两阶段
        with open(labels_dir_two / f"{img_name}.txt", 'w') as f:
            for det in two_dets:
                x1, y1, x2, y2 = det['box']
                cls = det['cls']
                # 转换为YOLO格式
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def generate_report(self, results: Dict, stats: Dict, 
                       single_avg_acc: float, two_avg_acc: float) -> str:
        """生成评估报告"""
        improvement = two_avg_acc - single_avg_acc
        
        report = f"""
{'='*60}
两阶段级联检测 - 评估报告
{'='*60}

【总体指标】

单阶段YOLO:
  - 平均数量准确率: {single_avg_acc:.2f}%
  - 总检测数: {stats['single_stage']['total_detections']}
  - 总GT数: {stats['single_stage']['total_gt']}

两阶段级联:
  - 平均数量准确率: {two_avg_acc:.2f}%
  - 总检测数: {stats['two_stage']['total_detections']}
  - 总GT数: {stats['two_stage']['total_gt']}

性能提升:
  - 准确率提升: {improvement:+.2f}%
  - {'✅ 两阶段更优' if improvement > 0 else '❌ 单阶段更优'}

{'='*60}

【详细分析】

"""
        
        # 找出提升最大的图像
        improvements = [(r['image'], r['improvement']) 
                       for r in results['comparison']]
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        report += "提升最大的前5张图像:\n"
        for i, (img_name, imp) in enumerate(improvements[:5], 1):
            report += f"  {i}. {img_name}: {imp*100:+.2f}%\n"
        
        report += "\n下降最大的5张图像:\n"
        for i, (img_name, imp) in enumerate(improvements[-5:], 1):
            report += f"  {i}. {img_name}: {imp*100:+.2f}%\n"
        
        report += f"\n{'='*60}\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description="两阶段级联检测 - 批量评估")
    parser.add_argument('--yolo-model', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--classifier', type=str, required=True, help='分类器权重路径')
    parser.add_argument('--data-yaml', type=str, required=True, help='数据集YAML配置')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], 
                       help='数据集划分')
    parser.add_argument('--model-type', type=str, default='mobilenet',
                       choices=['mlp', 'mobilenet'], help='分类器类型')
    parser.add_argument('--input-size', type=int, default=112, help='分类器输入尺寸')
    parser.add_argument('--num-classes', type=int, default=2, help='类别数（包括背景）')
    parser.add_argument('--imgsz', type=int, default=1280, help='推理尺寸')
    
    # 阈值参数
    parser.add_argument('--stage1-conf', type=float, default=0.05, 
                       help='第一阶段置信度阈值')
    parser.add_argument('--stage2-threshold', type=float, default=0.5,
                       help='第二阶段分类阈值')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                       help='单阶段YOLO置信度（用于对比）')
    
    # NMS参数
    parser.add_argument('--cross-class-nms', action='store_true', default=True,
                       help='启用跨类别NMS')
    parser.add_argument('--no-cross-class-nms', action='store_false', dest='cross_class_nms',
                       help='禁用跨类别NMS')
    parser.add_argument('--nms-iou', type=float, default=0.3,
                       help='跨类别NMS的IOU阈值')
    
    parser.add_argument('--save-dir', type=str, default='runs/cascaded_eval',
                       help='保存目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CascadedEvaluator(
        args.yolo_model,
        args.classifier,
        classifier_type=args.model_type,
        input_size=args.input_size,
        num_classes=args.num_classes,
        device=args.device,
        cross_class_nms=args.cross_class_nms,
        nms_iou=args.nms_iou
    )
    
    # 运行评估
    evaluator.evaluate_dataset(
        args.data_yaml,
        split=args.split,
        stage1_conf=args.stage1_conf,
        stage2_threshold=args.stage2_threshold,
        yolo_conf=args.yolo_conf,
        imgsz=args.imgsz,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()

