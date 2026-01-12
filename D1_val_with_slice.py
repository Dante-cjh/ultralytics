#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D1 数据验证脚本 (单尺度版本)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

from ultralytics import YOLO
from ultralytics.data.split_yolo import split_trainval
from ultralytics.utils import LOGGER


class BalloonTrainingPipeline:
    """D1 数据验证流水线"""
    
    def __init__(
        self,
        data_root: str,
        slice_dir: str,
        model_path: str = "yolo11n.pt",
        project_name: str = "D1_yolo11n_slice"
    ):
        """
        初始化训练流水线
        
        Args:
            data_root (str): 原始 Balloon 数据根目录
            slice_dir (str): 切片后数据保存目录
            model_path (str): 模型名称或路径
            project_name (str): 训练项目名称
        """
        self.data_root = Path(data_root)
        self.slice_dir = Path(slice_dir)
        self.model_path = model_path
        self.project_name = project_name
        
        # 验证路径
        if not self.data_root.exists():
            raise ValueError(f"数据根目录不存在: {self.data_root}")
    
    def check_data_structure(self) -> bool:
        """检查数据目录结构"""
        LOGGER.info("🔍 检查数据目录结构...")
        
        required_dirs = [
            "images/train",
            "images/val",
            "labels/train", 
            "labels/val"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.data_root / dir_path
            if not full_path.exists():
                missing_dirs.append(str(full_path))
        
        if missing_dirs:
            LOGGER.error("❌ 数据目录结构不完整")
            for missing_dir in missing_dirs:
                LOGGER.error(f"  缺少: {missing_dir}")
            return False
        
        LOGGER.info("✅ 数据目录结构检查通过")
        return True
    
    def val_model(
        self,
        name: str,
        imgsz: int = 640,
        batch: int = 16,
        device: int = 0,
    ) -> Optional[str]:
        """
        训练模型
        
        Args:
            epochs (int): 训练轮数
            imgsz (int): 输入图像尺寸
            batch (int): 批次大小
            device (int): GPU 设备编号
            patience (int): 早停耐心值
            resume (bool): 是否恢复训练
        
        Returns:
            str: 最佳模型路径
        """
        LOGGER.info("🚀 开始模型训练...")
        
        # 创建临时数据集配置文件，使用实际的切片目录
        import yaml
        import tempfile
        
        dataset_config = {
            'path': str(self.slice_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,
            'names': {0: 'hole', 1: 'cave', 2: 'unknow'}
        }
        
        # 保存临时yaml文件
        temp_yaml = Path(tempfile.gettempdir()) / f"{self.project_name}_data.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump(dataset_config, f)
        
        dataset_yaml = str(temp_yaml)
        
        # 初始化模型
        model = YOLO(self.model_path)
        LOGGER.info(f"模型地址: {self.model_path}")
        
        try:
            # 开始训练
            metrics = model.val(
                data=str(dataset_yaml),
                imgsz=imgsz,
                batch=batch,
                device=device,
                name=name,
                # === 其他设置 ===
                workers=4,                # 数据加载线程数
                verbose=True,             # 详细输出
                plots=True,               # 生成训练图表
            )
            
            return None
            
        except Exception as e:
            LOGGER.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_pipeline(
        self,
        # 训练参数
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: int = 0,
        patience: int = 30,
        resume: bool = False
    ) -> bool:
        """
        运行完整的训练流水线
        
        Returns:
            bool: 是否成功
        """
        LOGGER.info("🎯 开始 Balloon 完整训练流水线")
        LOGGER.info(f"项目名称: {self.project_name}")
        LOGGER.info(f"原始数据: {self.data_root}")
        LOGGER.info(f"切片数据: {self.slice_dir}")
        
        # 1. 检查数据结构
        if not self.check_data_structure():
            return False
        
        # 3. 模型训练
        best_model = self.val_model(imgsz, batch, device)
        
        LOGGER.info("🎉 完整训练流水线执行成功！")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="D1 数据验证脚本 (单尺度版本)")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, 
                       default="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format",
                       help="原始 Balloon 数据根目录")
    parser.add_argument("--slice-dir", type=str, 
                       default="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format_slice",
                       help="切片后数据保存目录")
    parser.add_argument("--project-name", type=str, default="D1_yolo11l_slice", help="训练项目名称")
    
    # 验证参数
    parser.add_argument("--model-path", type=str, default="/public/home/baichen/download/dcu_yolo/ultralytics/runs/detect/D1_yolov8l_slice_20251029_174115/weights/best.pt", help="模型名称或路径")
    parser.add_argument("--name", type=str, default="yolov8l_slice_vaildation_run", help="保存路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备编号")
    
    
    args = parser.parse_args()
    
    try:
        # 创建训练流水线
        pipeline = BalloonTrainingPipeline(
            model_path=args.model_path,
            data_root=args.data_root,
            slice_dir=args.slice_dir,
            project_name=args.project_name
        )

        pipeline.val_model(
            name=args.name,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
            
    except Exception as e:
        LOGGER.error(f"❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

