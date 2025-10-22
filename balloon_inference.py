#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Balloon 数据集普通推理脚本
使用训练好的模型对图像进行标准推理（非切片）
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List

from ultralytics import YOLO
from ultralytics.utils import LOGGER


class BalloonInference:
    """Balloon 数据集普通推理类"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cuda:0"
    ):
        """
        初始化推理器
        
        Args:
            model_path (str): 训练好的模型路径
            confidence_threshold (float): 置信度阈值
            iou_threshold (float): NMS IoU阈值
            device (str): 设备 ('cuda:0' 或 'cpu')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
        # 验证模型文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        LOGGER.info(f"🔍 加载模型: {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """加载 YOLO 模型"""
        self.model = YOLO(str(self.model_path))
        LOGGER.info(f"✅ 模型加载成功")
    
    def predict_image(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
        save_txt: bool = True,
        save_conf: bool = True,
        visualize: bool = True,
    ) -> dict:
        """
        对单张图像进行推理
        
        Args:
            image_path (str): 图像路径
            save_dir (str, optional): 保存结果的目录
            save_txt (bool): 是否保存txt格式结果
            save_conf (bool): 是否保存置信度
            visualize (bool): 是否保存可视化结果
        
        Returns:
            dict: 包含检测结果的字典
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        LOGGER.info(f"📸 处理图像: {image_path.name}")
        
        # 执行推理
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=visualize,
            save_txt=save_txt,
            save_conf=save_conf,
            project=save_dir if save_dir else 'runs/predict',
            name='',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        
        # 获取检测结果
        result = results[0]
        num_detections = len(result.boxes)
        
        LOGGER.info(f"   检测到 {num_detections} 个目标")
        
        # 返回结果信息
        return {
            "image_path": str(image_path),
            "num_detections": num_detections,
            "result": result,
        }
    
    def predict_directory(
        self,
        image_dir: str,
        save_dir: str = "runs/predict",
        save_txt: bool = True,
        save_conf: bool = True,
        visualize: bool = True,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[dict]:
        """
        对目录中所有图像进行批量推理
        
        Args:
            image_dir (str): 图像目录
            save_dir (str): 保存结果的目录
            save_txt (bool): 是否保存txt格式结果
            save_conf (bool): 是否保存置信度
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
        LOGGER.info(f"   置信度阈值: {self.confidence_threshold}")
        LOGGER.info(f"   IoU阈值: {self.iou_threshold}")
        
        # 批量推理（YOLO支持批量处理，更高效）
        results = self.model.predict(
            source=str(image_dir),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=visualize,
            save_txt=save_txt,
            save_conf=save_conf,
            project=save_dir,
            name='',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        
        # 统计结果
        result_list = []
        for result in results:
            num_detections = len(result.boxes)
            result_list.append({
                "image_path": result.path,
                "num_detections": num_detections,
                "result": result,
            })
        
        # 统计总结
        total_detections = sum(r["num_detections"] for r in result_list)
        LOGGER.info(f"\n🎉 批量推理完成！")
        LOGGER.info(f"   处理图像: {len(result_list)}")
        LOGGER.info(f"   总检测数: {total_detections}")
        LOGGER.info(f"   平均每张: {total_detections/len(result_list):.1f} 个目标")
        if visualize:
            LOGGER.info(f"   结果保存: {save_dir}")
        
        return result_list
    
    def validate(
        self,
        data_yaml: str,
        batch: int = 32,
        imgsz: int = 640,
        save_dir: str = "runs/val",
        name: str = "val",
    ) -> dict:
        """
        在验证集上评估模型
        
        Args:
            data_yaml (str): 数据集配置文件路径
            batch (int): 批次大小
            imgsz (int): 图像尺寸
            save_dir (str): 保存结果的目录
        
        Returns:
            dict: 验证结果
        """
        LOGGER.info(f"🔍 在验证集上评估模型...")
        LOGGER.info(f"   数据配置: {data_yaml}")
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        results = self.model.val(
            data=data_yaml,
            batch=batch,
            imgsz=imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save_json=True,
            plots=True,
            project=save_dir,
            name=name,
            exist_ok=True,
            save_dir=os.path.join(save_dir, name),
        )
        
        LOGGER.info(f"✅ 验证完成!")
        LOGGER.info(f"   mAP@0.5: {results.box.map50:.4f}")
        LOGGER.info(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        LOGGER.info(f"   Precision: {results.box.mp:.4f}")
        LOGGER.info(f"   Recall: {results.box.mr:.4f}")
        
        return {
            "map50": results.box.map50,
            "map": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Balloon 数据集普通推理脚本")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU阈值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 (cuda:0 或 cpu)")
    
    # 输入输出
    parser.add_argument("--source", type=str, help="图像路径或目录")
    parser.add_argument("--save-dir", type=str, default="runs/predict", help="结果保存目录")
    parser.add_argument("--no-visualize", action="store_true", help="不保存可视化结果")
    parser.add_argument("--no-save-txt", action="store_true", help="不保存txt格式结果")
    parser.add_argument("--no-save-conf", action="store_true", help="不保存置信度")
    
    # 验证模式
    parser.add_argument("--val", action="store_true", help="验证模式（需要提供--data）")
    parser.add_argument("--data", type=str, help="数据集配置文件（验证模式）")
    parser.add_argument("--batch", type=int, default=32, help="批次大小（验证模式）")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸（验证模式）")
    parser.add_argument("--name", type=str, default="val", help="验证结果目录名称")
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        LOGGER.info("🚀 初始化推理器...")
        inferencer = BalloonInference(
            model_path=args.model,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            device=args.device,
        )
        
        # 验证模式
        if args.val:
            if not args.data:
                LOGGER.error("❌ 验证模式需要提供 --data 参数")
                return
            
            inferencer.validate(
                data_yaml=args.data,
                batch=args.batch,
                imgsz=args.imgsz,
                save_dir=args.save_dir,
                name=args.name,
            )
            return
        
        # 推理模式
        if not args.source:
            LOGGER.error("❌ 推理模式需要提供 --source 参数")
            return
            
        source_path = Path(args.source)
        visualize = not args.no_visualize
        save_txt = not args.no_save_txt
        save_conf = not args.no_save_conf
        
        if source_path.is_file():
            # 单张图像推理
            result = inferencer.predict_image(
                image_path=str(source_path),
                save_dir=args.save_dir,
                save_txt=save_txt,
                save_conf=save_conf,
                visualize=visualize,
            )
            LOGGER.info(f"\n✅ 推理完成！")
            
        elif source_path.is_dir():
            # 批量推理
            results = inferencer.predict_directory(
                image_dir=str(source_path),
                save_dir=args.save_dir,
                save_txt=save_txt,
                save_conf=save_conf,
                visualize=visualize,
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

