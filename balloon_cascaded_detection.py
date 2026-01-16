#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段级联检测系统 - Balloon数据集版本

第一阶段：使用YOLO模型生成候选框（低置信度）
第二阶段：对候选框进行重分类（使用轻量级分类器）

使用方法:
1. 准备数据: python balloon_cascaded_detection.py prepare --yolo-model <path> --conf 0.05
2. 训练分类器: python balloon_cascaded_detection.py train --data-dir <path>
3. 推理: python balloon_cascaded_detection.py infer --yolo-model <path> --classifier <path> --image <path>
"""

import os
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ==================== 工具函数 ====================

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的IOU
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IOU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def assign_labels(proposals: List[Tuple], gt_boxes: List[Tuple], iou_threshold: float = 0.5) -> List[Dict]:
    """
    为候选框分配标签（类似于Faster R-CNN的assigner）
    
    Args:
        proposals: 候选框列表 [(cls, conf, x1, y1, x2, y2), ...]
        gt_boxes: GT框列表 [(cls, x1, y1, x2, y2), ...]
        iou_threshold: IOU阈值
    
    Returns:
        标注后的候选框 [{'box': [x1,y1,x2,y2], 'pred_cls': int, 'true_cls': int, 'is_positive': bool}, ...]
        - is_positive=True: 正样本（与GT匹配）
        - is_positive=False: 负样本（背景）
        - true_cls: GT类别，-1表示背景
    """
    labeled_proposals = []
    
    for proposal in proposals:
        if len(proposal) == 6:
            pred_cls, conf, x1, y1, x2, y2 = proposal
        else:
            pred_cls, x1, y1, x2, y2 = proposal[:5]
            conf = 1.0
        
        prop_box = np.array([x1, y1, x2, y2])
        
        # 寻找最佳匹配的GT
        max_iou = 0.0
        matched_gt_cls = -1  # -1表示背景
        
        for gt in gt_boxes:
            gt_cls = int(gt[0])
            gt_box = np.array([gt[1], gt[2], gt[3], gt[4]])
            
            iou = calculate_iou(prop_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_gt_cls = gt_cls
        
        # 判断是否为正样本
        is_positive = max_iou >= iou_threshold
        
        labeled_proposals.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'pred_cls': int(pred_cls),
            'conf': float(conf),
            'true_cls': matched_gt_cls if is_positive else -1,  # -1表示背景
            'is_positive': is_positive,
            'iou': float(max_iou)
        })
    
    return labeled_proposals


def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> List[Tuple]:
    """解析YOLO格式标签文件"""
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                boxes.append((cls, x1, y1, x2, y2))
    
    return boxes


def cross_class_nms(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    跨类别NMS：处理不同类别预测同一目标的情况
    
    策略：对于高度重叠的框（即使类别不同），只保留置信度最高的
    
    Args:
        detections: 检测结果列表 [{'box': [x1,y1,x2,y2], 'cls': int, 'conf': float}, ...]
        iou_threshold: IOU阈值，高于此值的框会被抑制
    
    Returns:
        NMS后的检测结果
    """
    if len(detections) == 0:
        return detections
    
    # 转换为numpy数组
    boxes = np.array([d['box'] for d in detections])
    scores = np.array([d['conf'] for d in detections])
    
    # 计算所有框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度排序（从高到低）
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # 保留当前置信度最高的框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # 计算当前框与其他框的IOU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IOU小于阈值的框
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [detections[i] for i in keep]


# ==================== 数据准备 ====================

class CascadedDataPreparer:
    """第一阶段数据准备器"""
    
    def __init__(self, yolo_model_path: str, conf_threshold: float = 0.05, 
                 iou_threshold: float = 0.5, device: str = "cuda:0"):
        """
        初始化数据准备器
        
        Args:
            yolo_model_path: YOLO模型路径
            conf_threshold: 第一阶段置信度阈值（低阈值以获得更多候选框）
            iou_threshold: 与GT匹配的IOU阈值
            device: 设备
        """
        self.model = YOLO(yolo_model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        print(f"✅ 加载YOLO模型: {yolo_model_path}")
        print(f"   置信度阈值: {conf_threshold}, IOU阈值: {iou_threshold}")
    
    def generate_proposals(self, image_path: str, imgsz: int = 1280) -> List[Tuple]:
        """
        使用YOLO生成候选框
        
        Args:
            image_path: 图像路径
            imgsz: 推理尺寸
        
        Returns:
            候选框列表 [(cls, conf, x1, y1, x2, y2), ...]
        """
        results = self.model.predict(
            source=image_path,
            imgsz=imgsz,
            conf=self.conf_threshold,
            iou=0.45,  # NMS阈值
            device=self.device,
            verbose=False,
            save=False,
        )
        
        result = results[0]
        proposals = []
        
        if len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            
            for i in range(len(xyxy)):
                proposals.append((
                    int(cls[i]),
                    float(conf[i]),
                    float(xyxy[i][0]),
                    float(xyxy[i][1]),
                    float(xyxy[i][2]),
                    float(xyxy[i][3])
                ))
        
        return proposals
    
    def prepare_dataset(self, data_yaml: str, split: str = "train", 
                       output_dir: str = "cascaded_data", imgsz: int = 1280,
                       force: bool = False, negative_ratio: float = 2.0,
                       balance_samples: bool = True):
        """
        准备训练数据集
        
        Args:
            data_yaml: 数据集YAML配置文件
            split: 'train' 或 'val'
            output_dir: 输出目录
            imgsz: 推理尺寸
            force: 是否强制重新生成（如果为False且数据已存在，则跳过）
            negative_ratio: 负样本与正样本的比例（默认2.0，即负样本数=正样本数*2）
            balance_samples: 是否平衡正负样本（下采样多数类）
        """
        # 创建输出目录
        output_path = Path(output_dir) / split
        
        # 检查是否已存在数据
        if not force and output_path.exists():
            list_file = output_path / 'data_list.json'
            stats_file = output_path / 'stats.json'
            
            if list_file.exists() and stats_file.exists():
                print(f"\n⏭️  {split}集数据已存在，跳过准备步骤")
                print(f"   数据路径: {output_path}")
                print(f"   如需重新生成，请使用 --force 参数")
                
                # 读取并显示统计信息
                import json
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                print(f"   总图像数: {stats['total_images']}")
                print(f"   总候选框数: {stats['total_proposals']}")
                print(f"   正样本数: {stats['positive_samples']}")
                print(f"   负样本数: {stats['negative_samples']}")
                return
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
        
        # 创建输出目录
        output_path = Path(output_dir) / split
        crops_dir = output_path / 'crops'
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        stats = {
            'total_images': 0,
            'total_proposals': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'class_dist': {}
        }
        
        # 准备数据列表
        data_list = []
        
        # 处理每张图像
        image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        
        print(f"\n🔍 处理 {split} 集...")
        for img_path in tqdm(image_files, desc=f"准备{split}数据"):
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            stats['total_images'] += 1
            
            # 生成候选框
            proposals = self.generate_proposals(str(img_path), imgsz)
            
            if len(proposals) == 0:
                continue
            
            # 读取GT
            label_path = label_dir / (img_path.stem + '.txt')
            gt_boxes = parse_yolo_label(str(label_path), img_w, img_h)
            
            # 分配标签
            labeled_proposals = assign_labels(proposals, gt_boxes, self.iou_threshold)
            
            # 保存每个候选框
            for idx, prop in enumerate(labeled_proposals):
                x1, y1, x2, y2 = [int(v) for v in prop['box']]
                
                # 裁剪区域（带边界检查）
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # 保存裁剪图像
                crop_name = f"{img_path.stem}_{idx}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                
                # 记录数据（确保类型可JSON序列化）
                data_list.append({
                    'crop_path': str(crop_path.relative_to(output_path)),
                    'true_cls': int(prop['true_cls']),
                    'pred_cls': int(prop['pred_cls']),
                    'conf': float(prop['conf']),
                    'iou': float(prop['iou']),
                    'is_positive': bool(prop['is_positive'])
                })
                
                # 统计
                stats['total_proposals'] += 1
                if prop['is_positive']:
                    stats['positive_samples'] += 1
                    cls = prop['true_cls']
                    stats['class_dist'][cls] = stats['class_dist'].get(cls, 0) + 1
                else:
                    stats['negative_samples'] += 1
        
        # 样本平衡（如果启用）
        if balance_samples and stats['positive_samples'] > 0 and stats['negative_samples'] > 0:
            print(f"\n⚖️  正负样本平衡...")
            print(f"   原始 - 正样本: {stats['positive_samples']}, 负样本: {stats['negative_samples']}")
            print(f"   目标负样本比例: {negative_ratio}:1")
            
            target_negative = int(stats['positive_samples'] * negative_ratio)
            current_ratio = stats['negative_samples'] / stats['positive_samples']
            
            import random
            random.seed(42)  # 固定随机种子以保证可复现
            
            # 分离正负样本
            positive_samples = [s for s in data_list if s['is_positive']]
            negative_samples = [s for s in data_list if not s['is_positive']]
            
            if stats['negative_samples'] > target_negative:
                # 情况1: 负样本过多 → 下采样负样本
                print(f"   📉 负样本过多，下采样负样本: {stats['negative_samples']} → {target_negative}")
                
                # 随机选择负样本
                sampled_negatives = random.sample(negative_samples, target_negative)
                
                # 合并数据
                data_list = positive_samples + sampled_negatives
                
                # 更新统计
                stats['negative_samples'] = target_negative
                stats['total_proposals'] = len(data_list)
                
                print(f"   ✅ 平衡后 - 正样本: {stats['positive_samples']}, 负样本: {stats['negative_samples']}")
                print(f"   负样本比例: {stats['negative_samples']/stats['positive_samples']:.2f}:1")
            
            elif stats['negative_samples'] < target_negative:
                # 情况2: 负样本过少（正样本过多）→ 下采样正样本
                print(f"   📉 负样本不足，下采样正样本以达到平衡")
                print(f"   当前负样本比例: {current_ratio:.2f}:1 (目标: {negative_ratio}:1)")
                
                # 计算目标正样本数
                target_positive = int(stats['negative_samples'] / negative_ratio)
                
                print(f"   下采样正样本: {stats['positive_samples']} → {target_positive}")
                
                # 随机选择正样本
                sampled_positives = random.sample(positive_samples, target_positive)
                
                # 合并数据
                data_list = sampled_positives + negative_samples
                
                # 更新统计
                stats['positive_samples'] = target_positive
                stats['total_proposals'] = len(data_list)
                
                print(f"   ✅ 平衡后 - 正样本: {stats['positive_samples']}, 负样本: {stats['negative_samples']}")
                print(f"   负样本比例: {stats['negative_samples']/stats['positive_samples']:.2f}:1")
            
            else:
                # 情况3: 负样本数量已经合理 → 不处理
                print(f"   ✅ 负样本数量已经合理，无需调整")
                print(f"   当前负样本比例: {current_ratio:.2f}:1 (目标: {negative_ratio}:1)")
        
        # 保存数据列表
        import json
        list_file = output_path / 'data_list.json'
        with open(list_file, 'w') as f:
            json.dump(data_list, f, indent=2)
        
        # 保存统计信息
        stats['negative_ratio'] = negative_ratio if balance_samples else None
        stats['balanced'] = balance_samples
        stats_file = output_path / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ {split}集准备完成!")
        print(f"   总图像数: {stats['total_images']}")
        print(f"   总候选框数: {stats['total_proposals']}")
        print(f"   正样本数: {stats['positive_samples']}")
        print(f"   负样本数: {stats['negative_samples']}")
        print(f"   正负比例: 1:{stats['negative_samples']/max(stats['positive_samples'],1):.2f}")
        print(f"   类别分布: {stats['class_dist']}")
        print(f"   数据保存至: {output_path}")


# ==================== 分类器模型 ====================

class SimpleMLP(nn.Module):
    """简单的MLP分类器"""
    
    def __init__(self, input_size: int = 112, num_classes: int = 2, 
                 hidden_dims: List[int] = [256, 128]):
        """
        Args:
            input_size: 输入图像大小
            num_classes: 类别数（包括背景）
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        # 简单的卷积特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # MLP分类头
        feature_dim = 128 * 4 * 4
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetClassifier(nn.Module):
    """基于MobileNetV2的分类器"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5):
        """
        Args:
            num_classes: 类别数（包括背景）
            pretrained: 是否使用预训练权重
            dropout: Dropout比例（默认0.5）
        """
        super().__init__()
        
        # 检查本地预训练模型
        local_model_path = Path("pretrained_models/mobilenet_v2-b0353104.pth")
        
        if pretrained and local_model_path.exists():
            # 从本地加载预训练模型
            print(f"   📦 从本地加载MobileNetV2预训练模型: {local_model_path}")
            self.backbone = models.mobilenet_v2(pretrained=False)
            state_dict = torch.load(local_model_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
        else:
            # 从网络下载（需要联网）
            if pretrained:
                print(f"   📥 从网络下载MobileNetV2预训练模型（需要联网）...")
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # 替换分类头（增加Dropout和额外的全连接层）
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== Loss函数 ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for Hard Example Mining
    
    论文: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    
    用途：自动关注难分类样本，降低简单样本的权重
    
    Args:
        alpha: 类别权重，用于处理类别不平衡
               - 可以是float（所有类别统一权重）
               - 可以是list（每个类别不同权重）
        gamma: focusing parameter，控制难易样本的权重差异
               - gamma=0时退化为标准交叉熵
               - gamma越大，简单样本权重越低
               - 推荐值：2.0-5.0
    
    公式：
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
    其中 p_t 是正确类别的预测概率：
        - p_t 接近1（简单样本）→ (1-p_t)^γ 接近0 → loss很小
        - p_t 接近0（难样本）→ (1-p_t)^γ 接近1 → loss正常
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重，默认0.25
            gamma: focusing参数，默认2.0
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs: [N, C] 模型输出logits（未经过softmax）
            targets: [N] 类别标签
        
        Returns:
            loss: scalar
        """
        # 计算交叉熵loss（不做reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率
        p = torch.exp(-ce_loss)  # p_t: 正确类别的预测概率
        
        # 计算focal weight
        focal_weight = (1 - p) ** self.gamma
        
        # 计算focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== 数据集 ====================

class CascadedDataset(Dataset):
    """级联检测数据集"""
    
    def __init__(self, data_list_path: str, transform=None, num_classes: int = 2):
        """
        Args:
            data_list_path: 数据列表JSON文件路径
            transform: 图像变换
            num_classes: 类别数（不包括背景）
        """
        import json
        
        with open(data_list_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.root_dir = Path(data_list_path).parent
        self.transform = transform
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 读取图像
        img_path = self.root_dir / item['crop_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 标签：true_cls=-1表示背景(label=0)，其他类别从1开始
        true_cls = item['true_cls']
        if true_cls == -1:
            label = 0  # 背景
        else:
            label = true_cls + 1  # 前景类别从1开始
        
        return image, label


# ==================== 训练器 ====================

class CascadedTrainer:
    """级联检测训练器"""
    
    def __init__(self, model: nn.Module, device: str = "cuda:0"):
        self.model = model.to(device)
        self.device = device
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, lr: float = 0.001, save_dir: str = "runs/cascaded_train",
              weight_decay: float = 0.01, patience: int = 10, 
              loss_type: str = 'focal', focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        训练分类器
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            lr: 学习率
            save_dir: 保存目录
            weight_decay: 权重衰减（L2正则化）
            patience: 早停的耐心轮数
            loss_type: 损失函数类型 ('ce' 或 'focal')
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 优化器和损失函数（使用AdamW，带权重衰减）
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 选择损失函数
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"   使用Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"   使用Cross Entropy Loss")
        
        # 学习率调度器：先余弦退火，再根据验证集性能调整
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n🚀 开始训练分类器...")
        print(f"   训练样本数: {len(train_loader.dataset)}")
        print(f"   验证样本数: {len(val_loader.dataset)}")
        print(f"   训练轮数: {num_epochs}")
        print(f"   学习率: {lr}")
        print(f"   权重衰减: {weight_decay}")
        print(f"   早停耐心: {patience}轮")
        print(f"   损失函数: {loss_type}")
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{train_loss/train_total:.4f}",
                    'acc': f"{100.*train_correct/train_total:.2f}%"
                })
            
            # 更新学习率
            scheduler_cosine.step()
            
            # 验证阶段
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # 根据验证性能调整学习率
            scheduler_plateau.step(val_acc)
            
            print(f"\n   Epoch {epoch+1}: "
                  f"Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Train Acc={100.*train_correct/train_total:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # 保存最佳模型和早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path / 'best.pt')
                print(f"   ✅ 保存最佳模型 (Val Acc={val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"   ⚠️  验证准确率未提升 ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n🛑 早停! {patience}轮验证准确率未提升")
                    break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), save_path / f'epoch_{epoch+1}.pt')
        
        print(f"\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   模型保存至: {save_path}")
    
    def evaluate(self, data_loader: DataLoader, criterion) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy


# ==================== 两阶段推理器 ====================

class CascadedDetector:
    """两阶段级联检测器"""
    
    def __init__(self, yolo_model_path: str, classifier_path: str,
                 classifier_type: str = "mobilenet", input_size: int = 112,
                 num_classes: int = 2, conf_threshold: float = 0.05,
                 classifier_threshold: float = 0.5, device: str = "cuda:0",
                 cross_class_nms: bool = True, nms_iou: float = 0.3):
        """
        初始化检测器
        
        Args:
            yolo_model_path: YOLO模型路径
            classifier_path: 分类器权重路径
            classifier_type: 分类器类型 ('mlp' 或 'mobilenet')
            input_size: 分类器输入尺寸
            num_classes: 类别数（包括背景）
            conf_threshold: 第一阶段置信度阈值
            classifier_threshold: 第二阶段分类阈值
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
        
        self.conf_threshold = conf_threshold
        self.classifier_threshold = classifier_threshold
        self.device = device
        self.use_cross_class_nms = cross_class_nms
        self.nms_iou = nms_iou
        
        print(f"✅ 加载两阶段检测器")
        print(f"   YOLO模型: {yolo_model_path}")
        print(f"   分类器: {classifier_path} (类型: {classifier_type})")
        print(f"   跨类别NMS: {'启用' if cross_class_nms else '禁用'} (IOU={nms_iou})")
    
    def detect(self, image_path: str, imgsz: int = 1280) -> List[Dict]:
        """
        两阶段检测
        
        Args:
            image_path: 图像路径
            imgsz: YOLO推理尺寸
        
        Returns:
            检测结果 [{'box': [x1,y1,x2,y2], 'cls': int, 'conf': float, 'stage1_cls': int, 'stage1_conf': float}, ...]
        """
        # 读取图像
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        
        # 第一阶段：YOLO生成候选框
        results = self.yolo_model.predict(
            source=image_path,
            imgsz=imgsz,
            conf=self.conf_threshold,
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
            if stage2_cls == 0 or stage2_conf < self.classifier_threshold:
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


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="两阶段级联检测系统")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 准备数据
    prepare_parser = subparsers.add_parser('prepare', help='准备训练数据')
    prepare_parser.add_argument('--yolo-model', type=str, required=True, help='YOLO模型路径')
    prepare_parser.add_argument('--data-yaml', type=str, default='/home/cjh/mmdetection/data/balloon/yolo_format/data.yaml', 
                               help='数据集YAML配置文件')
    prepare_parser.add_argument('--conf', type=float, default=0.05, help='第一阶段置信度阈值')
    prepare_parser.add_argument('--iou', type=float, default=0.5, help='与GT匹配的IOU阈值')
    prepare_parser.add_argument('--output-dir', type=str, default='cascaded_data', help='输出目录')
    prepare_parser.add_argument('--imgsz', type=int, default=1280, help='推理尺寸')
    prepare_parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    prepare_parser.add_argument('--force', action='store_true', help='强制重新生成数据（即使已存在）')
    prepare_parser.add_argument('--negative-ratio', type=float, default=2.0, 
                               help='负样本与正样本的比例（默认2.0，即负:正=2:1）')
    prepare_parser.add_argument('--no-balance', action='store_true', 
                               help='不进行样本平衡（保留所有样本）')
    
    # 训练分类器
    train_parser = subparsers.add_parser('train', help='训练分类器')
    train_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    train_parser.add_argument('--model-type', type=str, default='mobilenet', 
                             choices=['mlp', 'mobilenet'], help='模型类型')
    train_parser.add_argument('--input-size', type=int, default=112, help='输入图像大小')
    train_parser.add_argument('--num-classes', type=int, default=2, help='类别数（包括背景）')
    train_parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--save-dir', type=str, default='runs/cascaded_train', help='保存目录')
    train_parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    # Focal Loss参数
    train_parser.add_argument('--loss-type', type=str, default='focal', 
                             choices=['ce', 'focal'], help='损失函数类型: ce (CrossEntropy) 或 focal (FocalLoss)')
    train_parser.add_argument('--focal-alpha', type=float, default=0.25,
                             help='Focal Loss的alpha参数（类别权重）')
    train_parser.add_argument('--focal-gamma', type=float, default=2.0,
                             help='Focal Loss的gamma参数（难易样本权重差异，推荐2.0-5.0）')
    
    # 推理
    infer_parser = subparsers.add_parser('infer', help='两阶段推理')
    infer_parser.add_argument('--yolo-model', type=str, required=True, help='YOLO模型路径')
    infer_parser.add_argument('--classifier', type=str, required=True, help='分类器权重路径')
    infer_parser.add_argument('--model-type', type=str, default='mobilenet',
                             choices=['mlp', 'mobilenet'], help='分类器类型')
    infer_parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    infer_parser.add_argument('--imgsz', type=int, default=1280, help='YOLO推理尺寸')
    infer_parser.add_argument('--input-size', type=int, default=112, help='分类器输入尺寸')
    infer_parser.add_argument('--num-classes', type=int, default=2, help='类别数（包括背景）')
    infer_parser.add_argument('--conf', type=float, default=0.05, help='第一阶段置信度阈值')
    infer_parser.add_argument('--cls-threshold', type=float, default=0.5, help='第二阶段分类阈值')
    infer_parser.add_argument('--cross-class-nms', action='store_true', default=True, help='启用跨类别NMS')
    infer_parser.add_argument('--no-cross-class-nms', action='store_false', dest='cross_class_nms', help='禁用跨类别NMS')
    infer_parser.add_argument('--nms-iou', type=float, default=0.3, help='跨类别NMS的IOU阈值')
    infer_parser.add_argument('--save-dir', type=str, default='runs/cascaded_infer', help='保存目录')
    infer_parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        # 准备数据
        preparer = CascadedDataPreparer(
            args.yolo_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
        balance_samples = not args.no_balance
        preparer.prepare_dataset(
            args.data_yaml, 'train', args.output_dir, args.imgsz, 
            args.force, args.negative_ratio, balance_samples
        )
        preparer.prepare_dataset(
            args.data_yaml, 'val', args.output_dir, args.imgsz, 
            args.force, args.negative_ratio, balance_samples
        )
        
    elif args.command == 'train':
        # 创建数据加载器（增强的数据增强）
        train_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            
            # 几何变换
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),  # 随机旋转±15度
            
            # 颜色增强
            transforms.ColorJitter(
                brightness=0.2,  # 亮度
                contrast=0.2,    # 对比度
                saturation=0.2,  # 饱和度
                hue=0.1          # 色调
            ),
            
            transforms.ToTensor(),
            
            # 随机擦除（模拟遮挡）
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = CascadedDataset(
            f"{args.data_dir}/train/data_list.json",
            transform=train_transform,
            num_classes=args.num_classes - 1  # 减去背景
        )
        
        val_dataset = CascadedDataset(
            f"{args.data_dir}/val/data_list.json",
            transform=val_transform,
            num_classes=args.num_classes - 1
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
        
        # 创建模型（使用更高的dropout）
        if args.model_type == 'mlp':
            model = SimpleMLP(args.input_size, args.num_classes)
        else:
            model = MobileNetClassifier(args.num_classes, dropout=0.5)
        
        # 训练（使用权重衰减、早停和Focal Loss）
        trainer = CascadedTrainer(model, args.device)
        trainer.train(train_loader, val_loader, args.epochs, args.lr, args.save_dir,
                     weight_decay=0.01, patience=10,
                     loss_type=args.loss_type, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma)
        
    elif args.command == 'infer':
        # 两阶段推理
        detector = CascadedDetector(
            args.yolo_model,
            args.classifier,
            classifier_type=args.model_type,
            input_size=args.input_size,
            num_classes=args.num_classes,
            conf_threshold=args.conf,
            classifier_threshold=args.cls_threshold,
            device=args.device,
            cross_class_nms=args.cross_class_nms,
            nms_iou=args.nms_iou
        )
        
        detections = detector.detect(args.image, args.imgsz)
        
        print(f"\n✅ 检测完成，共检测到 {len(detections)} 个目标")
        for i, det in enumerate(detections):
            print(f"   {i+1}. 类别={det['cls']}, 置信度={det['conf']:.3f}, "
                  f"框={det['box']}, "
                  f"[Stage1: cls={det['stage1_cls']}, conf={det['stage1_conf']:.3f}]")
        
        # 可视化
        img = cv2.imread(args.image)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls{det['cls']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # 保存结果
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / f"{Path(args.image).stem}_cascaded.jpg"
        cv2.imwrite(str(output_file), img)
        print(f"   结果保存至: {output_file}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

