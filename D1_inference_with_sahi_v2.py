#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D1 数据集 SAHI 切片推理脚本
使用训练好的模型对大尺寸图像进行切片推理
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

from ultralytics.utils import LOGGER

# 检查SAHI版本
try:
    import sahi
    LOGGER.info(f"📦 SAHI版本: {sahi.__version__}")
except ImportError:
    LOGGER.error("❌ SAHI未安装，请运行: pip install sahi")
    exit(1)


class BalloonSAHIInference:
    """D1 数据集 SAHI 切片推理类"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: str = "cuda:0"
    ):
        """
        初始化 SAHI 推理器
        
        Args:
            model_path (str): 训练好的模型路径
            confidence_threshold (float): 置信度阈值
            device (str): 设备 ('cuda:0' 或 'cpu')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.detection_model = None
        
        # 验证模型文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            # SAHI 0.11.14 使用 yolov8 作为 model_type
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=str(self.model_path),
                confidence_threshold=self.confidence_threshold,
                device=self.device,
            )
            LOGGER.info(f"✅ 模型加载成功")
            LOGGER.info(f"   模型路径: {self.model_path}")
            LOGGER.info(f"   置信度阈值: {self.confidence_threshold}")
            LOGGER.info(f"   设备: {self.device}")
        except Exception as e:
            LOGGER.error(f"❌ 模型加载失败: {e}")
            raise
    
    def predict_image(
        self,
        image_path: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.15,
        overlap_width_ratio: float = 0.15,
        postprocess_type: str = "NMS",
        postprocess_threshold: float = 0.5,
        postprocess_metric: str = "IOS",
        save_dir: Optional[str] = None,
        visualize: bool = True,
        save_txt: bool = True,
        save_conf: bool = True,
        min_box_area: int = 100,  # 最小检测框面积
        max_detections: int = 100,  # 最大检测数量
    ) -> dict:
        """
        对单张图像进行切片推理
        
        Args:
            image_path (str): 图像路径
            slice_height (int): 切片高度
            slice_width (int): 切片宽度
            overlap_height_ratio (float): 高度重叠比例 (0.0-1.0)
            overlap_width_ratio (float): 宽度重叠比例 (0.0-1.0)
            save_dir (str, optional): 保存可视化结果的目录
            visualize (bool): 是否保存可视化结果
        
        Returns:
            dict: 包含检测结果的字典
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        LOGGER.info(f"📸 处理图像: {image_path.name}")
        
        # 读取图像
        image = read_image(str(image_path))
        h, w = image.shape[:2]
        LOGGER.info(f"   图像尺寸: {w}x{h}")
        
        # 执行切片推理
        try:
            LOGGER.info(f"   开始SAHI切片推理...")
            LOGGER.info(f"   切片参数: {slice_width}x{slice_height}, 重叠: {overlap_width_ratio:.1%}x{overlap_height_ratio:.1%}")
            
            result = get_sliced_prediction(
                image,
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_type=postprocess_type,  # 使用NMS去除重复检测框
                postprocess_match_metric=postprocess_metric,  # 使用IOS匹配指标
                postprocess_match_threshold=postprocess_threshold,  # NMS IoU阈值
                postprocess_class_agnostic=False,  # 类别感知的NMS
            )
            LOGGER.info(f"   SAHI推理完成")
            LOGGER.info(f"   原始检测数量: {len(result.object_prediction_list)}")
        except Exception as e:
            LOGGER.error(f"   ❌ SAHI推理失败: {e}")
            raise
        
        # 应用额外的过滤逻辑
        filtered_predictions = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = bbox
            box_area = (x2 - x1) * (y2 - y1)
            
            # 过滤条件
            if box_area >= min_box_area:
                filtered_predictions.append(pred)
        
        # 按置信度排序并限制数量
        filtered_predictions.sort(key=lambda x: x.score.value, reverse=True)
        if len(filtered_predictions) > max_detections:
            filtered_predictions = filtered_predictions[:max_detections]
            LOGGER.info(f"   检测框数量限制: {len(result.object_prediction_list)} -> {max_detections}")
        
        # 更新结果
        result.object_prediction_list = filtered_predictions
        
        # 统计检测结果
        num_detections = len(result.object_prediction_list)
        LOGGER.info(f"   检测到 {num_detections} 个目标 (过滤后)")
        
        # 调试信息：打印检测结果详情
        if num_detections > 0:
            LOGGER.info(f"   检测详情:")
            # 按置信度排序显示
            sorted_predictions = sorted(result.object_prediction_list, 
                                      key=lambda x: x.score.value, reverse=True)
            for i, pred in enumerate(sorted_predictions[:5]):  # 只显示前5个
                bbox = pred.bbox.to_xyxy()
                LOGGER.info(f"     [{i+1}] {pred.category.name}: {pred.score.value:.3f} "
                           f"bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
            if num_detections > 5:
                LOGGER.info(f"     ... 还有 {num_detections - 5} 个检测结果")
            
            # 统计置信度分布
            confidences = [pred.score.value for pred in result.object_prediction_list]
            LOGGER.info(f"   置信度统计: 最高={max(confidences):.3f}, 最低={min(confidences):.3f}, 平均={sum(confidences)/len(confidences):.3f}")
        
        # 保存可视化结果
        if visualize and save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 手动绘制检测框（修复版本）
            vis_image = image.copy()
            img_h, img_w = vis_image.shape[:2]
            
            for pred in result.object_prediction_list:
                bbox = pred.bbox.to_xyxy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # 边界检查：确保坐标在图像范围内
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                # 确保边界框有效
                if x2 > x1 and y2 > y1:
                    # 绘制边界框（绿色）
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签和置信度
                    label = f"{pred.category.name}: {pred.score.value:.2f}"
                    
                    # 计算标签背景大小
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 计算标签位置，确保不超出图像边界
                    label_x = x1
                    label_y = y1 - 5  # 标签在边界框上方
                    
                    # 如果标签会超出图像顶部，则放在边界框内部
                    if label_y - label_h < 0:
                        label_y = y1 + label_h + 5
                    
                    # 确保标签不超出图像边界
                    label_x = max(0, min(label_x, img_w - label_w))
                    label_y = max(label_h, min(label_y, img_h))
                    
                    # 绘制标签背景
                    cv2.rectangle(
                        vis_image, 
                        (label_x, label_y - label_h - baseline), 
                        (label_x + label_w, label_y), 
                        (0, 255, 0), 
                        -1
                    )
                    
                    # 绘制标签文字（黑色）
                    cv2.putText(
                        vis_image, label, (label_x, label_y - baseline), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                    )
                else:
                    LOGGER.warning(f"   跳过无效边界框: ({x1}, {y1}, {x2}, {y2})")
            
            # 保存图像
            output_path = save_path / f"{image_path.stem}_visual.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            LOGGER.info(f"   可视化结果保存到: {output_path}")
        
        # 保存txt格式标签
        if save_txt and save_dir:
            save_path = Path(save_dir)
            labels_dir = save_path / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            # 生成YOLO格式的txt标签文件
            txt_path = labels_dir / f"{image_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for pred in result.object_prediction_list:
                    bbox = pred.bbox.to_xyxy()
                    x1, y1, x2, y2 = bbox

                    x_center = (x1 + x2) / 2.0 / w
                    y_center = (y1 + y2) / 2.0 / h
                    box_width = (x2-x1) / w
                    box_height = (y2 - y1) / h

                    # 获取类别ID
                    class_id = pred.category.id

                    # 写入格式：class_id x_center y_center width height [confidence]
                    if save_conf:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {pred.score.value:.6f}\n")
                    else:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    
                LOGGER.info(f"   标签文件保存到: {txt_path}")

        # 返回结果信息
        return {
            "image_path": str(image_path),
            "image_size": (w, h),
            "num_detections": num_detections,
            "detections": result.object_prediction_list,
            "result": result,
        }
    
    def predict_directory(
        self,
        image_dir: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.15,
        overlap_width_ratio: float = 0.15,
        postprocess_type: str = "NMS",
        postprocess_threshold: float = 0.5,
        postprocess_metric: str = "IOS",
        save_dir: str = "runs/sahi_inference",
        visualize: bool = True,
        save_txt: bool = True,
        save_conf: bool = True,
        min_box_area: int = 100,
        max_detections: int = 100,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list:
        """
        对目录中所有图像进行批量推理
        
        Args:
            image_dir (str): 图像目录
            slice_height (int): 切片高度
            slice_width (int): 切片宽度
            overlap_height_ratio (float): 高度重叠比例
            overlap_width_ratio (float): 宽度重叠比例
            save_dir (str): 保存结果的目录
            visualize (bool): 是否保存可视化结果
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
        
        LOGGER.info(f"🎯 开始批量推理，共 {len(image_files)} 张图像")
        LOGGER.info(f"   切片参数: {slice_width}x{slice_height}, 重叠: {overlap_width_ratio:.1%}x{overlap_height_ratio:.1%}")
        
        # 处理每张图像
        results = []
        for i, image_path in enumerate(image_files, 1):
            LOGGER.info(f"[{i}/{len(image_files)}]")
            try:
                result = self.predict_image(
                    image_path=str(image_path),
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    postprocess_type=postprocess_type,
                    postprocess_threshold=postprocess_threshold,
                    postprocess_metric=postprocess_metric,
                    save_dir=save_dir,
                    visualize=visualize,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    min_box_area=min_box_area,
                    max_detections=max_detections,
                )
                results.append(result)
            except Exception as e:
                LOGGER.error(f"   ❌ 处理失败: {e}")
        
        # 统计总结
        total_detections = sum(r["num_detections"] for r in results)
        LOGGER.info(f"\n🎉 批量推理完成！")
        LOGGER.info(f"   处理图像: {len(results)}/{len(image_files)}")
        LOGGER.info(f"   总检测数: {total_detections}")
        LOGGER.info(f"   平均每张: {total_detections/len(results):.1f} 个目标")
        if visualize:
            LOGGER.info(f"   结果保存: {save_dir}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="D1 数据集 SAHI 切片推理脚本")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cuda:0 或 cpu)")
    
    # 输入输出
    parser.add_argument("--source", type=str, required=True, help="图像路径或目录")
    parser.add_argument("--save-dir", type=str, default="runs/sahi_inference", help="结果保存目录")
    parser.add_argument("--no-visualize", action="store_true", help="不保存可视化结果")
    parser.add_argument("--no-save-txt", action="store_true", help="不保存txt格式标签")
    parser.add_argument("--no-save-conf", action="store_true", help="不保存置信度")
    
    # 切片参数
    parser.add_argument("--slice-height", type=int, default=640, help="切片高度")
    parser.add_argument("--slice-width", type=int, default=640, help="切片宽度")
    parser.add_argument("--overlap-height", type=float, default=0.15, help="高度重叠比例 (0.0-1.0)")
    parser.add_argument("--overlap-width", type=float, default=0.15, help="宽度重叠比例 (0.0-1.0)")
    
    # 后处理参数
    parser.add_argument("--postprocess-type", type=str, default="NMS", choices=["NMS", "NMM"], help="后处理方法")
    parser.add_argument("--postprocess-threshold", type=float, default=0.5, help="NMS/NMM阈值")
    parser.add_argument("--postprocess-metric", type=str, default="IOS", choices=["IOS", "IOU"], help="匹配指标")
    
    # 高级过滤参数
    parser.add_argument("--min-box-area", type=int, default=100, help="最小检测框面积")
    parser.add_argument("--max-detections", type=int, default=100, help="最大检测数量")
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        LOGGER.info("🚀 初始化 SAHI 推理器...")
        inferencer = BalloonSAHIInference(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
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
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height,
                overlap_width_ratio=args.overlap_width,
                postprocess_type=args.postprocess_type,
                postprocess_threshold=args.postprocess_threshold,
                postprocess_metric=args.postprocess_metric,
                save_dir=args.save_dir,
                visualize=visualize,
                save_txt=save_txt,
                save_conf=save_conf,
                min_box_area=args.min_box_area,
                max_detections=args.max_detections,
            )
            LOGGER.info(f"\n✅ 推理完成！")
            
        elif source_path.is_dir():
            # 批量推理
            results = inferencer.predict_directory(
                image_dir=str(source_path),
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height,
                overlap_width_ratio=args.overlap_width,
                postprocess_type=args.postprocess_type,
                postprocess_threshold=args.postprocess_threshold,
                postprocess_metric=args.postprocess_metric,
                save_dir=args.save_dir,
                visualize=visualize,
                save_txt=save_txt,
                save_conf=save_conf,
                min_box_area=args.min_box_area,
                max_detections=args.max_detections,
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

