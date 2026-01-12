#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多尺度推理脚本
使用多个图像尺度进行推理，然后融合所有尺度的检测结果
支持NMS和WBF两种融合方式
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def cross_class_nms(boxes, scores, classes, iou_threshold=0.5):
    """
    跨类别NMS：对所有类别的检测框进行NMS，去除重复检测
    用于解决多尺度融合时同一个目标被多个类别检测的问题
    
    Args:
        boxes (np.ndarray): 检测框 [N, 4] (x1, y1, x2, y2) 归一化坐标
        scores (np.ndarray): 置信度 [N]
        classes (np.ndarray): 类别 [N]
        iou_threshold (float): IoU阈值
    
    Returns:
        boxes, scores, classes: 过滤后的检测结果
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    
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
    
    # 返回保留的检测结果
    return boxes[keep], scores[keep], classes[keep]


class MultiScaleInference:
    """多尺度推理类"""
    
    def __init__(
        self,
        model_path: str,
        scales: List[int] = [640, 832, 1024, 1280],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cuda:0",
        fusion_method: str = "nms",  # 'nms' 或 'wbf'
        class_agnostic_nms: bool = True,  # 是否使用跨类别NMS
    ):
        """
        初始化多尺度推理器
        
        Args:
            model_path (str): 训练好的模型路径
            scales (List[int]): 多个推理尺度，例如 [640, 832, 1024, 1280]
            confidence_threshold (float): 置信度阈值
            iou_threshold (float): NMS/WBF IoU阈值
            device (str): 设备 ('cuda:0' 或 'cpu')
            fusion_method (str): 融合方法 'nms' 或 'wbf'
            class_agnostic_nms (bool): 是否使用跨类别NMS（默认True，解决多标签重复问题）
        """
        self.model_path = Path(model_path)
        self.scales = sorted(scales)  # 按尺度排序
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.fusion_method = fusion_method.lower()
        self.class_agnostic_nms = class_agnostic_nms
        self.model = None
        
        # 验证模型文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 检查WBF依赖
        if self.fusion_method == "wbf":
            try:
                from ensemble_boxes import weighted_boxes_fusion
                self.wbf = weighted_boxes_fusion
                LOGGER.info("✅ WBF (Weighted Boxes Fusion) 已启用")
            except ImportError:
                LOGGER.warning("⚠️ ensemble-boxes未安装，回退到NMS")
                LOGGER.warning("   安装命令: pip install ensemble-boxes")
                self.fusion_method = "nms"
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        LOGGER.info(f"   推理尺度: {self.scales}")
        LOGGER.info(f"   融合方法: {self.fusion_method.upper()}")
        LOGGER.info(f"   跨类别NMS: {'启用' if self.class_agnostic_nms else '禁用'}")
        self._load_model()
    
    def _load_model(self):
        """加载 YOLO 模型"""
        self.model = YOLO(str(self.model_path))
        LOGGER.info(f"✅ 模型加载成功")
    
    def _predict_single_scale(
        self,
        image: np.ndarray,
        scale: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在单个尺度上进行推理
        
        Args:
            image (np.ndarray): 输入图像 (H, W, C)
            scale (int): 推理尺度
        
        Returns:
            boxes (np.ndarray): 检测框 [N, 4] (x1, y1, x2, y2) 归一化坐标
            scores (np.ndarray): 置信度 [N]
            classes (np.ndarray): 类别 [N]
        """
        # 执行推理
        results = self.model.predict(
            source=image,
            imgsz=scale,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            save=False,
        )
        
        result = results[0]
        
        # 提取检测结果
        if len(result.boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        scores = result.boxes.conf.cpu().numpy()  # [N]
        classes = result.boxes.cls.cpu().numpy()  # [N]
        
        # 转换为归一化坐标 (0-1)
        h, w = image.shape[:2]
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= w  # x坐标归一化
        boxes_norm[:, [1, 3]] /= h  # y坐标归一化
        
        return boxes_norm, scores, classes
    
    def _nms_fusion(
        self,
        all_boxes: List[np.ndarray],
        all_scores: List[np.ndarray],
        all_classes: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用NMS融合多尺度检测结果
        
        Args:
            all_boxes: 所有尺度的检测框列表
            all_scores: 所有尺度的置信度列表
            all_classes: 所有尺度的类别列表
        
        Returns:
            boxes, scores, classes: 融合后的结果
        """
        # 合并所有尺度的检测结果
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        if len(boxes) == 0:
            return boxes, scores, classes
        
        num_before = len(boxes)
        
        # 使用跨类别NMS
        if self.class_agnostic_nms:
            boxes, scores, classes = cross_class_nms(
                boxes, scores, classes, self.iou_threshold
            )
            LOGGER.info(f"   跨类别NMS: {num_before} -> {len(boxes)}")
        else:
            # 按类别NMS（原始方法）
            # 转换为torch tensor进行NMS
            boxes_tensor = torch.from_numpy(boxes).float()
            scores_tensor = torch.from_numpy(scores).float()
            classes_tensor = torch.from_numpy(classes).long()
            
            # 对每个类别分别执行NMS
            keep_indices = []
            unique_classes = torch.unique(classes_tensor)
            
            for cls in unique_classes:
                cls_mask = classes_tensor == cls
                cls_boxes = boxes_tensor[cls_mask]
                cls_scores = scores_tensor[cls_mask]
                
                # 执行NMS
                keep = torch.ops.torchvision.nms(
                    cls_boxes,
                    cls_scores,
                    self.iou_threshold
                )
                
                # 获取原始索引
                cls_indices = torch.where(cls_mask)[0]
                keep_indices.extend(cls_indices[keep].tolist())
            
            # 保留NMS后的检测结果
            keep_indices = sorted(keep_indices)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]
            
            LOGGER.info(f"   按类别NMS: {num_before} -> {len(boxes)}")
        
        return boxes, scores, classes
    
    def _wbf_fusion(
        self,
        all_boxes: List[np.ndarray],
        all_scores: List[np.ndarray],
        all_classes: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用WBF (Weighted Boxes Fusion) 融合多尺度检测结果
        
        WBF相比NMS的优势:
        - 不是简单删除重叠框，而是融合它们
        - 使用加权平均来确定最终框的位置
        - 通常比NMS有更好的定位精度
        
        Args:
            all_boxes: 所有尺度的检测框列表
            all_scores: 所有尺度的置信度列表  
            all_classes: 所有尺度的类别列表
        
        Returns:
            boxes, scores, classes: 融合后的结果
        """
        # WBF需要的格式: list of [x1, y1, x2, y2] (归一化坐标)
        boxes_list = [boxes.tolist() for boxes in all_boxes]
        scores_list = [scores.tolist() for scores in all_scores]
        labels_list = [classes.astype(int).tolist() for classes in all_classes]
        
        # 执行WBF
        boxes, scores, labels = self.wbf(
            boxes_list,
            scores_list,
            labels_list,
            weights=None,  # 所有尺度权重相同
            iou_thr=self.iou_threshold,
            skip_box_thr=self.confidence_threshold,
        )
        
        return np.array(boxes), np.array(scores), np.array(labels)
    
    def predict_image(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
        visualize: bool = True,
        save_txt: bool = True,
        save_conf: bool = True,
    ) -> dict:
        """
        对单张图像进行多尺度推理
        
        Args:
            image_path (str): 图像路径
            save_dir (str, optional): 保存结果的目录
            visualize (bool): 是否保存可视化结果
            save_txt (bool): 是否保存txt格式结果
            save_conf (bool): 是否保存置信度
        
        Returns:
            dict: 包含检测结果的字典
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        LOGGER.info(f"📸 处理图像: {image_path.name}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        LOGGER.info(f"   原始尺寸: {w}x{h}")
        
        # 在多个尺度上进行推理
        all_boxes = []
        all_scores = []
        all_classes = []
        
        start_time = time.time()
        
        for scale in self.scales:
            boxes, scores, classes = self._predict_single_scale(image_rgb, scale)
            
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                LOGGER.info(f"   尺度 {scale}: 检测到 {len(boxes)} 个目标")
            else:
                LOGGER.info(f"   尺度 {scale}: 未检测到目标")
        
        # 融合多尺度结果
        if len(all_boxes) == 0:
            LOGGER.info(f"   ⚠️ 所有尺度均未检测到目标")
            boxes_fused = np.array([])
            scores_fused = np.array([])
            classes_fused = np.array([])
        else:
            LOGGER.info(f"   融合方法: {self.fusion_method.upper()}")
            if self.fusion_method == "wbf":
                boxes_fused, scores_fused, classes_fused = self._wbf_fusion(
                    all_boxes, all_scores, all_classes
                )
            else:  # nms
                boxes_fused, scores_fused, classes_fused = self._nms_fusion(
                    all_boxes, all_scores, all_classes
                )
        
        inference_time = time.time() - start_time
        num_detections = len(boxes_fused)
        
        LOGGER.info(f"   ✅ 融合后: {num_detections} 个目标 (耗时: {inference_time:.2f}s)")
        
        # 反归一化坐标
        if num_detections > 0:
            boxes_pixel = boxes_fused.copy()
            boxes_pixel[:, [0, 2]] *= w
            boxes_pixel[:, [1, 3]] *= h
        else:
            boxes_pixel = boxes_fused
        
        # 保存可视化结果
        if visualize and save_dir and num_detections > 0:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            vis_image = image.copy()
            
            for i, (box, score, cls) in enumerate(zip(boxes_pixel, scores_fused, classes_fused)):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"Class {int(cls)}: {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # 标签位置
                label_y = max(y1 - 5, label_h)
                cv2.rectangle(
                    vis_image,
                    (x1, label_y - label_h - baseline),
                    (x1 + label_w, label_y),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    vis_image, label, (x1, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
            
            # 保存图像
            output_path = save_path / f"{image_path.stem}_multiscale.jpg"
            cv2.imwrite(str(output_path), vis_image)
            LOGGER.info(f"   可视化结果: {output_path}")
        
        # 保存txt格式标签
        if save_txt and save_dir and num_detections > 0:
            save_path = Path(save_dir)
            labels_dir = save_path / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            txt_path = labels_dir / f"{image_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for box, score, cls in zip(boxes_fused, scores_fused, classes_fused):
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    if save_conf:
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {score:.6f}\n")
                    else:
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            LOGGER.info(f"   标签文件: {txt_path}")
        
        return {
            "image_path": str(image_path),
            "image_size": (w, h),
            "num_detections": num_detections,
            "boxes": boxes_pixel,
            "scores": scores_fused,
            "classes": classes_fused,
            "inference_time": inference_time,
        }
    
    def predict_directory(
        self,
        image_dir: str,
        save_dir: str = "runs/multiscale_inference",
        visualize: bool = True,
        save_txt: bool = True,
        save_conf: bool = True,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[dict]:
        """
        对目录中所有图像进行批量多尺度推理
        
        Args:
            image_dir (str): 图像目录
            save_dir (str): 保存结果的目录
            visualize (bool): 是否保存可视化结果
            save_txt (bool): 是否保存txt格式结果
            save_conf (bool): 是否保存置信度
            image_extensions (tuple): 支持的图像扩展名
        
        Returns:
            list: 所有图像的检测结果列表
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            LOGGER.warning(f"⚠️ 目录中未找到图像文件: {image_dir}")
            return []
        
        LOGGER.info(f"🎯 开始批量多尺度推理")
        LOGGER.info(f"   图像数量: {len(image_files)}")
        LOGGER.info(f"   推理尺度: {self.scales}")
        LOGGER.info(f"   融合方法: {self.fusion_method.upper()}")
        
        # 处理每张图像
        results = []
        total_start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            LOGGER.info(f"\n[{i}/{len(image_files)}]")
            try:
                result = self.predict_image(
                    image_path=str(image_path),
                    save_dir=save_dir,
                    visualize=visualize,
                    save_txt=save_txt,
                    save_conf=save_conf,
                )
                results.append(result)
            except Exception as e:
                LOGGER.error(f"   ❌ 处理失败: {e}")
        
        total_time = time.time() - total_start_time
        
        # 统计总结
        if results:
            total_detections = sum(r["num_detections"] for r in results)
            avg_time = total_time / len(results)
            
            LOGGER.info(f"\n{'='*70}")
            LOGGER.info(f"🎉 批量推理完成！")
            LOGGER.info(f"{'='*70}")
            LOGGER.info(f"   处理图像: {len(results)}/{len(image_files)}")
            LOGGER.info(f"   总检测数: {total_detections}")
            LOGGER.info(f"   平均每张: {total_detections/len(results):.1f} 个目标")
            LOGGER.info(f"   总耗时: {total_time:.2f}s")
            LOGGER.info(f"   平均耗时: {avg_time:.2f}s/张")
            if visualize:
                LOGGER.info(f"   结果保存: {save_dir}")
            LOGGER.info(f"{'='*70}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多尺度推理脚本")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--scales", type=int, nargs="+", default=[640, 832, 1024, 1280],
                       help="推理尺度列表，例如: --scales 640 832 1024 1280")
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS/WBF IoU阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cuda:0 或 cpu)")
    parser.add_argument("--fusion", type=str, default="nms", choices=["nms", "wbf"],
                       help="融合方法: nms 或 wbf (Weighted Boxes Fusion)")
    parser.add_argument("--no-cross-class-nms", action="store_true", 
                       help="禁用跨类别NMS（默认启用，解决多标签重复问题）")
    
    # 输入输出
    parser.add_argument("--source", type=str, required=True, help="图像路径或目录")
    parser.add_argument("--save-dir", type=str, default="runs/multiscale_inference", help="结果保存目录")
    parser.add_argument("--no-visualize", action="store_true", help="不保存可视化结果")
    parser.add_argument("--no-save-txt", action="store_true", help="不保存txt格式结果")
    parser.add_argument("--no-save-conf", action="store_true", help="不保存置信度")
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        LOGGER.info("🚀 初始化多尺度推理器...")
        class_agnostic_nms = not args.no_cross_class_nms
        inferencer = MultiScaleInference(
            model_path=args.model,
            scales=args.scales,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            device=args.device,
            fusion_method=args.fusion,
            class_agnostic_nms=class_agnostic_nms,
        )
        
        # 判断输入是文件还是目录
        source_path = Path(args.source)
        visualize = not args.no_visualize
        save_txt = not args.no_save_txt
        save_conf = not args.no_save_conf
        
        if source_path.is_file():
            # 单张图像推理
            result = inferencer.predict_image(
                image_path=str(source_path),
                save_dir=args.save_dir,
                visualize=visualize,
                save_txt=save_txt,
                save_conf=save_conf,
            )
            LOGGER.info(f"\n✅ 推理完成！")
            
        elif source_path.is_dir():
            # 批量推理
            results = inferencer.predict_directory(
                image_dir=str(source_path),
                save_dir=args.save_dir,
                visualize=visualize,
                save_txt=save_txt,
                save_conf=save_conf,
            )
        else:
            LOGGER.error(f"❌ 无效的输入路径: {source_path}")
            return
        
    except Exception as e:
        LOGGER.error(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

