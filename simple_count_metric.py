#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义检测数量对比Metric - 简化版本
实现公式: 1 - |pred_count - true_count| / true_count

作者: AI Assistant
日期: 2025-01-29
"""

import pickle
import numpy as np
from pathlib import Path


class SimpleCountMetric:
    """简化的检测数量对比Metric"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置"""
        self.pred_counts = []
        self.true_counts = []
        self.image_names = []
    
    def update(self, pred_count: int, true_count: int, image_name: str = ""):
        """更新metric"""
        self.pred_counts.append(pred_count)
        self.true_counts.append(true_count)
        self.image_names.append(image_name)
    
    def compute_metric(self, pred_count: int, true_count: int) -> float:
        """计算单个metric值"""
        if true_count > 0:
            return 1 - abs(pred_count - true_count) / true_count
        else:
            return 1.0 if pred_count == 0 else float('-inf')
    
    def get_results(self):
        """获取结果"""
        if not self.pred_counts:
            return {"error": "没有数据"}
        
        # 计算所有metric值
        metric_values = []
        for pred, true in zip(self.pred_counts, self.true_counts):
            metric_values.append(self.compute_metric(pred, true))
        
        # 过滤有效值
        valid_values = [v for v in metric_values if v != float('-inf')]
        
        return {
            "total_images": len(self.pred_counts),
            "valid_images": len(valid_values),
            "total_pred_boxes": sum(self.pred_counts),
            "total_true_boxes": sum(self.true_counts),
            "mean_metric": np.mean(valid_values) if valid_values else 0.0,
            "min_metric": np.min(valid_values) if valid_values else 0.0,
            "max_metric": np.max(valid_values) if valid_values else 0.0,
            "perfect_matches": sum(1 for v in valid_values if v == 1.0),
            "good_predictions": sum(1 for v in valid_values if v >= 0.8),
            "poor_predictions": sum(1 for v in valid_values if v < 0.5),
            "detailed_results": [
                {
                    "image_name": name,
                    "pred_count": pred,
                    "true_count": true,
                    "metric_value": metric
                }
                for name, pred, true, metric in zip(
                    self.image_names, self.pred_counts, self.true_counts, metric_values
                )
            ]
        }


def load_ground_truth(labels_dir: str) -> dict:
    """加载真实标签"""
    labels_dir = Path(labels_dir)
    ground_truth = {}
    
    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem + ".jpg"
        with open(label_file, 'r') as f:
            lines = f.readlines()
        true_count = len([line for line in lines if line.strip()])
        ground_truth[image_name] = true_count
    
    return ground_truth


def evaluate_from_pkl(pkl_path: str, labels_dir: str):
    """从PKL文件评估"""
    print(f"📊 从PKL文件评估: {pkl_path}")
    
    # 加载PKL文件
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 加载真实标签
    ground_truth = load_ground_truth(labels_dir)
    
    # 创建metric
    metric = SimpleCountMetric()
    
    # 处理数据
    if "results" in pkl_data:
        results = pkl_data["results"]
    else:
        results = [pkl_data]
    
    for result in results:
        image_name = result["image_name"]
        pred_count = result["num_detections"]
        true_count = ground_truth.get(image_name, 0)
        metric.update(pred_count, true_count, image_name)
    
    return metric.get_results()


def main():
    """主函数"""
    print("🔬 自定义检测数量对比Metric")
    print("公式: 1 - |pred_count - true_count| / true_count")
    print("="*60)
    
    # 测试公式
    print("🧪 公式测试:")
    test_cases = [
        (5, 5, "完美匹配"),
        (4, 5, "少预测1个"),
        (6, 5, "多预测1个"),
        (0, 5, "完全漏检"),
        (10, 5, "多预测5个"),
        (15, 5, "多预测10个"),
    ]
    
    metric = SimpleCountMetric()
    for pred, true, desc in test_cases:
        metric_value = metric.compute_metric(pred, true)
        print(f"   预测={pred:2d}, 真实={true:2d}, Metric={metric_value:6.3f}, {desc}")
    
    print("\n🎈 Balloon数据集评估:")
    
    # 评估参数
    pkl_path = "/public/home/baichen/download/dcu_yolo/ultralytics/runs/inference_pkl/D1_yolo11m_inference_pkl_results/all_inference_results.pkl"
    labels_dir = "/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format/labels/val"
    
    if Path(pkl_path).exists():
        results = evaluate_from_pkl(pkl_path, labels_dir)
        
        print(f"📈 评估结果:")
        print(f"   总图像数: {results['total_images']}")
        print(f"   有效图像数: {results['valid_images']}")
        print(f"   总预测框数: {results['total_pred_boxes']}")
        print(f"   总真实框数: {results['total_true_boxes']}")
        print(f"   平均Metric值: {results['mean_metric']:.4f}")
        print(f"   Metric范围: [{results['min_metric']:.4f}, {results['max_metric']:.4f}]")
        print(f"   完美匹配: {results['perfect_matches']} 张")
        print(f"   良好预测: {results['good_predictions']} 张")
        print(f"   较差预测: {results['poor_predictions']} 张")
        
        print(f"\n📸 前5个详细结果:")
        for i, detail in enumerate(results['detailed_results'][:5]):
            print(f"   {i+1}. {detail['image_name']}: "
                  f"预测={detail['pred_count']}, 真实={detail['true_count']}, "
                  f"Metric={detail['metric_value']:.4f}")
        
        # 分析结果
        print(f"\n📊 结果分析:")
        avg_pred = results['total_pred_boxes'] / max(results['total_images'],1)
        avg_true = results['total_true_boxes'] / max(results['total_images'],1)
        print(f"   平均预测框数: {avg_pred:.2f}")
        print(f"   平均真实框数: {avg_true:.2f}")
        # print(f"   预测/真实比例: {avg_pred/avg_true:.2f}")
        
        if results['mean_metric'] < 0:
            print(f"   ⚠️  模型存在严重过检测问题")
        elif results['mean_metric'] < 0.5:
            print(f"   ⚠️  模型检测数量准确性较差")
        elif results['mean_metric'] < 0.8:
            print(f"   ✅ 模型检测数量准确性一般")
        else:
            print(f"   🎉 模型检测数量准确性很好")
    
    else:
        print(f"❌ PKL文件不存在: {pkl_path}")
    
    print(f"\n🎉 评估完成!")


if __name__ == "__main__":
    main()
