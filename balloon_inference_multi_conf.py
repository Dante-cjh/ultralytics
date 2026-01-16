#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多置信度推理脚本
测试不同置信度阈值对检测数量的影响
找到最优的置信度参数以达到最佳数量准确率

使用方法:
python balloon_inference_multi_conf.py \
    --model best.pt \
    --source /path/to/images \
    --true-labels /path/to/labels \
    --conf-list 0.05 0.1 0.15 0.2 0.25 0.3
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def count_lines_in_file(file_path: str) -> int:
    """计算txt文件中的非空行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return len([line for line in lines if line.strip()])
    except Exception:
        return 0


def evaluate_count_accuracy(
    pred_labels_dir: str,
    true_labels_dir: str,
) -> Dict:
    """
    评估检测数量准确率
    
    Args:
        pred_labels_dir: 预测标签目录
        true_labels_dir: 真实标签目录
    
    Returns:
        评估结果字典
    """
    pred_dir = Path(pred_labels_dir)
    true_dir = Path(true_labels_dir)
    
    # 获取共同文件
    pred_files = {f.stem for f in pred_dir.glob("*.txt")}
    true_files = {f.stem for f in true_dir.glob("*.txt")}
    common_files = pred_files & true_files
    
    if not common_files:
        return {"error": "没有共同文件", "global_metric": 0}
    
    total_true = 0
    total_pred = 0
    metrics = []
    
    for filename in common_files:
        true_count = count_lines_in_file(str(true_dir / f"{filename}.txt"))
        pred_count = count_lines_in_file(str(pred_dir / f"{filename}.txt"))
        
        total_true += true_count
        total_pred += pred_count
        
        if true_count > 0:
            metric = 1 - abs(pred_count - true_count) / true_count
        else:
            metric = 1.0 if pred_count == 0 else 0.0
        
        metrics.append(metric)
    
    # 计算全局指标
    if total_true > 0:
        global_metric = 1 - abs(total_pred - total_true) / total_true
    else:
        global_metric = 1.0 if total_pred == 0 else 0.0
    
    avg_metric = sum(metrics) / len(metrics) if metrics else 0
    
    return {
        "global_metric": global_metric,
        "avg_metric": avg_metric,
        "total_true": total_true,
        "total_pred": total_pred,
        "diff": total_pred - total_true,
        "num_files": len(common_files),
    }


class MultiConfInference:
    """多置信度推理类"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0"
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        self.model = YOLO(str(self.model_path))
        LOGGER.info("✅ 模型加载成功")
    
    def predict_with_conf(
        self,
        image_dir: str,
        save_dir: str,
        imgsz: int = 1280,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> int:
        """
        使用指定置信度进行推理
        
        Args:
            image_dir: 图像目录
            save_dir: 保存目录
            imgsz: 推理尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        
        Returns:
            总检测数量
        """
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        
        # 创建标签目录
        labels_dir = save_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像列表
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            LOGGER.warning(f"⚠️ 未找到图像: {image_dir}")
            return 0
        
        total_detections = 0
        
        # 批量推理
        results = self.model.predict(
            source=str(image_dir),
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
            save=False,
            save_txt=False,
        )
        
        # 保存结果
        for result in results:
            img_path = Path(result.path)
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            # 保存txt标签
            txt_path = labels_dir / f"{img_path.stem}.txt"
            
            if num_detections > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 获取图像尺寸
                h, w = result.orig_shape
                
                with open(txt_path, 'w') as f:
                    for box, score, cls in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2.0 / w
                        y_center = (y1 + y2) / 2.0 / h
                        box_width = (x2 - x1) / w
                        box_height = (y2 - y1) / h
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {score:.6f}\n")
            else:
                # 创建空文件
                txt_path.touch()
        
        return total_detections
    
    def test_multi_conf(
        self,
        image_dir: str,
        true_labels_dir: str,
        save_base_dir: str,
        imgsz: int = 1280,
        conf_list: List[float] = None,
        iou_threshold: float = 0.5,
    ) -> Dict:
        """
        测试多个置信度阈值
        
        Args:
            image_dir: 图像目录
            true_labels_dir: 真实标签目录
            save_base_dir: 保存基础目录
            imgsz: 推理尺寸
            conf_list: 置信度列表
            iou_threshold: NMS IoU阈值
        
        Returns:
            所有结果
        """
        if conf_list is None:
            conf_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        LOGGER.info(f"🎯 开始多置信度测试")
        LOGGER.info(f"   置信度列表: {conf_list}")
        LOGGER.info(f"   IoU阈值: {iou_threshold}")
        LOGGER.info(f"   图像尺寸: {imgsz}")
        
        results = []
        best_result = None
        best_metric = -float('inf')
        
        for i, conf in enumerate(conf_list, 1):
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info(f"[{i}/{len(conf_list)}] 测试置信度: {conf}")
            LOGGER.info(f"{'='*60}")
            
            # 创建保存目录
            save_dir = Path(save_base_dir) / f"conf_{conf}"
            
            # 推理
            start_time = time.time()
            total_detections = self.predict_with_conf(
                image_dir=image_dir,
                save_dir=str(save_dir),
                imgsz=imgsz,
                conf_threshold=conf,
                iou_threshold=iou_threshold,
            )
            inference_time = time.time() - start_time
            
            # 评估
            eval_result = evaluate_count_accuracy(
                str(save_dir / "labels"),
                true_labels_dir
            )
            
            result = {
                "conf": conf,
                "iou": iou_threshold,
                "total_detections": total_detections,
                "inference_time": inference_time,
                **eval_result
            }
            results.append(result)
            
            LOGGER.info(f"   总检测数: {total_detections}")
            LOGGER.info(f"   真实总数: {eval_result['total_true']}")
            LOGGER.info(f"   差值: {eval_result['diff']:+d}")
            LOGGER.info(f"   全局Metric: {eval_result['global_metric']:.4f} ({eval_result['global_metric']*100:.2f}%)")
            LOGGER.info(f"   平均Metric: {eval_result['avg_metric']:.4f}")
            LOGGER.info(f"   耗时: {inference_time:.2f}s")
            
            if eval_result['global_metric'] > best_metric:
                best_metric = eval_result['global_metric']
                best_result = result
        
        # 打印总结
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info(f"🏆 多置信度测试完成！")
        LOGGER.info(f"{'='*70}")
        
        LOGGER.info(f"\n📊 结果汇总表:")
        LOGGER.info(f"{'conf':>8} | {'预测数':>8} | {'真实数':>8} | {'差值':>8} | {'全局Metric':>12} | {'平均Metric':>12}")
        LOGGER.info(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")
        
        for r in results:
            LOGGER.info(f"{r['conf']:>8.2f} | {r['total_detections']:>8d} | {r['total_true']:>8d} | {r['diff']:>+8d} | {r['global_metric']:>12.4f} | {r['avg_metric']:>12.4f}")
        
        LOGGER.info(f"\n🏆 最佳参数:")
        LOGGER.info(f"   置信度: {best_result['conf']}")
        LOGGER.info(f"   全局Metric: {best_result['global_metric']:.4f} ({best_result['global_metric']*100:.2f}%)")
        LOGGER.info(f"   预测总数: {best_result['total_detections']}")
        LOGGER.info(f"   真实总数: {best_result['total_true']}")
        LOGGER.info(f"   差值: {best_result['diff']:+d}")
        
        # 保存结果到JSON
        save_path = Path(save_base_dir) / "multi_conf_results.json"
        with open(save_path, 'w') as f:
            json.dump({
                "best_conf": best_result['conf'],
                "best_metric": best_metric,
                "all_results": results
            }, f, indent=2)
        LOGGER.info(f"\n📁 结果已保存: {save_path}")
        
        return {
            "best_conf": best_result['conf'],
            "best_metric": best_metric,
            "best_result": best_result,
            "all_results": results
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多置信度推理测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 测试多个置信度:
   python balloon_inference_multi_conf.py \\
       --model best.pt \\
       --source /path/to/images \\
       --true-labels /path/to/labels \\
       --conf-list 0.05 0.1 0.15 0.2 0.25 0.3

2. 使用默认置信度列表:
   python balloon_inference_multi_conf.py \\
       --model best.pt \\
       --source /path/to/images \\
       --true-labels /path/to/labels

3. 指定NMS IoU阈值:
   python balloon_inference_multi_conf.py \\
       --model best.pt \\
       --source /path/to/images \\
       --true-labels /path/to/labels \\
       --iou 0.6
        """
    )
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸")
    
    # 输入输出
    parser.add_argument("--source", type=str, required=True, help="图像目录")
    parser.add_argument("--true-labels", type=str, required=True, help="真实标签目录")
    parser.add_argument("--save-dir", type=str, default="runs/multi_conf_test", help="保存目录")
    
    # 测试参数
    parser.add_argument("--conf-list", type=float, nargs="+", 
                       default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                       help="置信度列表")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU阈值")
    
    args = parser.parse_args()
    
    try:
        LOGGER.info("🚀 初始化多置信度推理器...")
        inferencer = MultiConfInference(args.model, args.device)
        
        results = inferencer.test_multi_conf(
            image_dir=args.source,
            true_labels_dir=args.true_labels,
            save_base_dir=args.save_dir,
            imgsz=args.imgsz,
            conf_list=args.conf_list,
            iou_threshold=args.iou,
        )
        
    except Exception as e:
        LOGGER.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

