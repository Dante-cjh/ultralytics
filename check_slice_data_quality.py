#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
切片数据质量检测脚本
检测切片后的数据集是否存在以下问题：
1. 图像文件缺少对应的标签文件（负样本问题）
2. 标签坐标是否正确归一化到[0, 1]范围
3. 标签中心点是否超出边界
4. 标签宽高是否异常（过大或过小）
5. 可视化部分样本以验证标签正确性
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict

from ultralytics.utils import LOGGER


class SliceDataQualityChecker:
    """切片数据质量检测器"""
    
    def __init__(self, data_root: str):
        """
        初始化检测器
        
        Args:
            data_root: 切片后的数据根目录
        """
        self.data_root = Path(data_root)
        self.issues = defaultdict(list)
        
        if not self.data_root.exists():
            raise ValueError(f"数据根目录不存在: {self.data_root}")
    
    def check_split(self, split: str = "train") -> Dict:
        """
        检查指定分割的数据质量
        
        Args:
            split: 数据分割 ('train' 或 'val')
        
        Returns:
            检测结果统计字典
        """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🔍 开始检查 {split} 数据集")
        LOGGER.info(f"{'='*60}")
        
        images_dir = self.data_root / "images" / split
        labels_dir = self.data_root / "labels" / split
        
        if not images_dir.exists():
            LOGGER.error(f"❌ 图像目录不存在: {images_dir}")
            return {}
        
        if not labels_dir.exists():
            LOGGER.error(f"❌ 标签目录不存在: {labels_dir}")
            return {}
        
        # 获取所有图像文件
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        LOGGER.info(f"📊 找到 {len(image_files)} 个图像文件")
        
        # 统计信息
        stats = {
            'total_images': len(image_files),
            'images_with_labels': 0,
            'images_without_labels': 0,
            'empty_labels': 0,
            'total_objects': 0,
            'invalid_coords': 0,
            'out_of_range_coords': 0,
            'abnormal_sizes': 0,
            'problematic_files': []
        }
        
        # 检查每个图像
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 检查1: 是否有对应的标签文件
            if not label_path.exists():
                stats['images_without_labels'] += 1
                self.issues['missing_label'].append(str(img_path.name))
                continue
            
            stats['images_with_labels'] += 1
            
            # 读取标签
            labels = self._read_label(label_path)
            
            # 检查2: 标签是否为空（负样本）
            if len(labels) == 0:
                stats['empty_labels'] += 1
                self.issues['empty_label'].append(str(img_path.name))
                # 空标签文件也要继续检查图像是否有效
                # continue  # 注释掉，让它继续检查图像
            else:
                stats['total_objects'] += len(labels)
            
            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                LOGGER.warning(f"⚠️ 无法读取图像: {img_path}")
                continue
            img_h, img_w = img.shape[:2]
            
            # 检查每个标签
            for i, label in enumerate(labels):
                cls_id, x_center, y_center, width, height = label
                
                # 检查3: 坐标是否在[0, 1]范围内
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 <= width <= 1 and 0 <= height <= 1):
                    stats['out_of_range_coords'] += 1
                    self.issues['out_of_range'].append(
                        f"{img_path.name}: label[{i}] = [{cls_id}, {x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f}]"
                    )
                
                # 检查4: 边界框是否有效（转换为绝对坐标）
                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h
                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h
                
                # 检查边界框是否超出图像范围
                if x1 < -1 or y1 < -1 or x2 > img_w + 1 or y2 > img_h + 1:
                    stats['invalid_coords'] += 1
                    self.issues['invalid_bbox'].append(
                        f"{img_path.name}: bbox[{i}] = ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), image_size=({img_w}, {img_h})"
                    )
                
                # 检查5: 宽高是否异常
                abs_width = width * img_w
                abs_height = height * img_h
                
                # 宽高过小（可能是裁剪错误）
                if abs_width < 5 or abs_height < 5:
                    stats['abnormal_sizes'] += 1
                    self.issues['too_small'].append(
                        f"{img_path.name}: label[{i}] size = ({abs_width:.1f}, {abs_height:.1f})"
                    )
                
                # 宽高过大（可能是归一化错误）
                if abs_width > img_w * 0.95 or abs_height > img_h * 0.95:
                    stats['abnormal_sizes'] += 1
                    self.issues['too_large'].append(
                        f"{img_path.name}: label[{i}] size = ({abs_width:.1f}, {abs_height:.1f}), image_size=({img_w}, {img_h})"
                    )
        
        # 输出统计结果
        self._print_stats(split, stats)
        
        return stats
    
    def _read_label(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """读取YOLO格式标签文件"""
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((cls_id, x_center, y_center, width, height))
        except Exception as e:
            LOGGER.error(f"读取标签文件失败 {label_path}: {e}")
        return labels
    
    def _print_stats(self, split: str, stats: Dict):
        """打印统计结果"""
        LOGGER.info(f"\n📊 {split.upper()} 数据集质量统计:")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"总图像数量:              {stats['total_images']}")
        LOGGER.info(f"有标签的图像:            {stats['images_with_labels']}")
        LOGGER.info(f"❌ 缺少标签文件的图像:   {stats['images_without_labels']}")
        LOGGER.info(f"空标签文件数量:          {stats['empty_labels']}")
        LOGGER.info(f"总目标数量:              {stats['total_objects']}")
        LOGGER.info(f"")
        LOGGER.info(f"⚠️ 问题统计:")
        LOGGER.info(f"  坐标超出[0,1]范围:     {stats['out_of_range_coords']}")
        LOGGER.info(f"  边界框超出图像范围:    {stats['invalid_coords']}")
        LOGGER.info(f"  异常尺寸的目标:        {stats['abnormal_sizes']}")
        
        # 计算负样本比例（空标签文件或缺失标签文件）
        if stats['total_images'] > 0:
            negative_samples = stats['empty_labels'] + stats['images_without_labels']
            positive_samples = stats['images_with_labels'] - stats['empty_labels']
            negative_ratio = negative_samples / stats['total_images']
            positive_ratio = positive_samples / stats['total_images']
            LOGGER.info(f"")
            LOGGER.info(f"📈 样本分布:")
            LOGGER.info(f"  正样本比例:            {positive_ratio:.2%} ({positive_samples})")
            LOGGER.info(f"  负样本比例:            {negative_ratio:.2%} ({negative_samples})")
            
            if negative_samples == 0:
                LOGGER.warning(f"")
                LOGGER.warning(f"⚠️⚠️⚠️ 警告: 没有负样本（空切片）！")
                LOGGER.warning(f"这可能导致模型过拟合，出现'到处都是目标'的爆框问题！")
    
    def print_issue_details(self, max_examples: int = 10):
        """打印问题详情"""
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🔍 问题详情 (每类最多显示{max_examples}个)")
        LOGGER.info(f"{'='*60}")
        
        if not self.issues:
            LOGGER.info("✅ 未发现数据质量问题！")
            return
        
        for issue_type, examples in self.issues.items():
            LOGGER.info(f"\n❌ {issue_type} (共{len(examples)}个):")
            for example in examples[:max_examples]:
                LOGGER.info(f"  - {example}")
            if len(examples) > max_examples:
                LOGGER.info(f"  ... 还有 {len(examples) - max_examples} 个")
    
    def visualize_samples(
        self, 
        split: str = "train", 
        num_samples: int = 5,
        save_dir: str = "runs/check_slice_quality"
    ):
        """
        可视化部分样本以验证标签正确性
        
        Args:
            split: 数据分割
            num_samples: 可视化样本数量
            save_dir: 保存目录
        """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"📸 可视化 {num_samples} 个样本")
        LOGGER.info(f"{'='*60}")
        
        images_dir = self.data_root / "images" / split
        labels_dir = self.data_root / "labels" / split
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 随机选择样本
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if len(image_files) == 0:
            LOGGER.warning("没有找到图像文件")
            return
        
        np.random.seed(42)
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for img_path in samples:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 读取标签
            if label_path.exists():
                labels = self._read_label(label_path)
                
                # 绘制边界框
                for cls_id, x_center, y_center, width, height in labels:
                    # 转换为绝对坐标
                    x1 = int((x_center - width / 2) * img_w)
                    y1 = int((y_center - height / 2) * img_h)
                    x2 = int((x_center + width / 2) * img_w)
                    y2 = int((y_center + height / 2) * img_h)
                    
                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制中心点
                    center_x = int(x_center * img_w)
                    center_y = int(y_center * img_h)
                    cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
                    
                    # 绘制标签
                    label_text = f"cls:{int(cls_id)}"
                    cv2.putText(img, label_text, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 添加标题
                title = f"{img_path.name} ({len(labels)} objects)"
            else:
                # 没有标签的图像（负样本）
                title = f"{img_path.name} (NO LABEL - Negative Sample)"
                cv2.putText(img, "NEGATIVE SAMPLE", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 添加标题
            cv2.putText(img, title, (10, img_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存图像
            output_path = save_path / f"{split}_{img_path.name}"
            cv2.imwrite(str(output_path), img)
        
        LOGGER.info(f"✅ 可视化结果保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="切片数据质量检测脚本")
    parser.add_argument("--data-root", type=str, required=True, help="切片后的数据根目录")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="要检查的数据分割")
    parser.add_argument("--visualize", action="store_true", help="可视化部分样本")
    parser.add_argument("--num-samples", type=int, default=10, help="可视化样本数量")
    parser.add_argument("--save-dir", type=str, default="runs/check_slice_quality", help="可视化结果保存目录")
    parser.add_argument("--show-details", action="store_true", help="显示问题详情")
    
    args = parser.parse_args()
    
    try:
        # 创建检测器
        checker = SliceDataQualityChecker(args.data_root)
        
        # 检查每个分割
        all_stats = {}
        for split in args.splits:
            stats = checker.check_split(split)
            all_stats[split] = stats
        
        # 显示问题详情
        if args.show_details:
            checker.print_issue_details()
        
        # 可视化样本
        if args.visualize:
            for split in args.splits:
                checker.visualize_samples(split, args.num_samples, args.save_dir)
        
        # 总结
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"✅ 数据质量检查完成！")
        LOGGER.info(f"{'='*60}")
        
        # 判断是否有严重问题
        has_serious_issues = False
        for split, stats in all_stats.items():
            negative_samples = stats.get('empty_labels', 0) + stats.get('images_without_labels', 0)
            if negative_samples == 0:
                LOGGER.warning(f"\n⚠️ {split} 数据集没有负样本，这可能导致训练问题！")
                has_serious_issues = True
            
            if stats.get('out_of_range_coords', 0) > 0:
                LOGGER.warning(f"\n⚠️ {split} 数据集有 {stats['out_of_range_coords']} 个坐标超出范围的标签！")
                has_serious_issues = True
        
        if has_serious_issues:
            LOGGER.warning(f"\n🔧 建议:")
            LOGGER.warning(f"  1. 检查 split_yolo.py 中的 crop_and_save 函数")
            LOGGER.warning(f"  2. 确保空切片被正确保存（图像+空标签文件）")
            LOGGER.warning(f"  3. 确保标签坐标正确归一化到切片窗口")
            LOGGER.warning(f"  4. 重新执行数据切片，使用修复后的代码")
        
    except Exception as e:
        LOGGER.error(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

