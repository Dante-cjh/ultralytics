#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D1 数据切片 + 训练集成脚本 (单尺度版本)
将数据切片和模型训练集成到一个完整的工作流中
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
    """Balloon 数据切片和训练流水线"""
    
    def __init__(
        self,
        data_root: str,
        slice_dir: str,
        model_name: str = "yolo11n.pt",
        project_name: str = "D1_yolo11n_slice"
    ):
        """
        初始化训练流水线
        
        Args:
            data_root (str): 原始 Balloon 数据根目录
            slice_dir (str): 切片后数据保存目录
            model_name (str): 模型名称或路径
            project_name (str): 训练项目名称
        """
        self.data_root = Path(data_root)
        self.slice_dir = Path(slice_dir)
        self.model_name = model_name
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
    
    def get_dataset_stats(self, data_path: Path) -> dict:
        """获取数据集统计信息"""
        stats = {}
        for split in ["train", "val"]:
            img_dir = data_path / "images" / split
            lbl_dir = data_path / "labels" / split
            
            img_count = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            
            stats[split] = {"images": img_count, "labels": lbl_count}
        
        return stats
    
    def slice_data(
        self,
        crop_size: int = 640,
        gap: int = 100,
        rates: Tuple[float, ...] = (1.0,),
        force_slice: bool = False
    ) -> bool:
        """
        执行数据切片
        
        Args:
            crop_size (int): 切片窗口大小
            gap (int): 窗口重叠大小
            rates (Tuple[float, ...]): 多尺度缩放比例
            force_slice (bool): 是否强制重新切片
        
        Returns:
            bool: 是否成功
        """
        LOGGER.info("📸 开始数据切片...")
        
        # 检查是否需要重新切片
        if self.slice_dir.exists() and not force_slice:
            slice_stats = self.get_dataset_stats(self.slice_dir)
            if slice_stats["train"]["images"] > 0:
                LOGGER.info(f"✅ 发现已切片的数据: {self.slice_dir}")
                LOGGER.info(f"   训练集: {slice_stats['train']['images']} 图像")
                LOGGER.info(f"   验证集: {slice_stats['val']['images']} 图像")
                LOGGER.info("   使用 --force-slice 强制重新切片")
                return True
        
        # 获取原始数据统计
        orig_stats = self.get_dataset_stats(self.data_root)
        LOGGER.info(f"原始数据统计:")
        for split, stats in orig_stats.items():
            LOGGER.info(f"  {split}: {stats['images']} 图像, {stats['labels']} 标签")
        
        LOGGER.info(f"切片参数:")
        LOGGER.info(f"  窗口大小: {crop_size}x{crop_size}")
        LOGGER.info(f"  重叠大小: {gap}")
        LOGGER.info(f"  缩放比例: {rates}")
        
        try:
            start_time = time.time()
            
            # 执行切片
            split_trainval(
                data_root=str(self.data_root),
                save_dir=str(self.slice_dir),
                crop_size=crop_size,
                gap=gap,
                rates=rates
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 获取切片后统计
            slice_stats = self.get_dataset_stats(self.slice_dir)
            LOGGER.info(f"✅ 数据切片完成 (耗时: {duration:.1f}s)")
            LOGGER.info(f"切片后数据统计:")
            for split, stats in slice_stats.items():
                LOGGER.info(f"  {split}: {stats['images']} 图像, {stats['labels']} 标签")
            
            return True
            
        except Exception as e:
            LOGGER.error(f"❌ 数据切片失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_model(
        self,
        epochs: int = 5,
        imgsz: int = 640,
        batch: int = 16,
        device: int = 0,
        patience: int = 30,
        resume: bool = False
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
        model = YOLO(self.model_name)
        LOGGER.info(f"模型: {self.model_name}")
        LOGGER.info(f"数据: {dataset_yaml} -> {self.slice_dir}")
        LOGGER.info(f"训练参数: epochs={epochs}, imgsz={imgsz}, batch={batch}, patience={patience}")
        
        try:
            # 开始训练
            results = model.train(
                data=str(dataset_yaml),

                # === 基础训练参数 ===
                epochs=epochs,            # 训练轮数
                imgsz=imgsz,              # 图像尺寸
                batch=batch,              # 批大小
                device=device,            # GPU设备
                
                # === 项目管理 ===
                project="runs/detect",
                name=f"{self.project_name}",
                resume=resume,
                exist_ok=True,            # 允许覆盖现有实验

                # === 早停和保存 ===
                patience=patience,        # 早停耐心值
                save_period=20,           # 每20轮保存一次

                # === 训练优化 ===
                amp=True,                 # 启用AMP
                cache=False,              # 不缓存图像
                rect=False,               # 不使用矩形训练
                cos_lr=True,              # 使用余弦学习率调度
                lr0=0.005,                 # 初始学习率
                lrf=0.02,                 # 最终学习率因子

                # === 其他设置 ===
                workers=4,                # 数据加载线程数
                verbose=True,             # 详细输出
                seed=42,                  # 随机种子，保证可重现性
                deterministic=True,       # 确定性训练
                single_cls=False,         # 单类别训练
                plots=True,               # 生成训练图表

                # === 数据增强设置 ===
                degrees=15.0,             # 旋转角度
                flipud=0.0,               # 竖直翻转
                fliplr=0.5,               # 水平翻转
                mosaic=1.0,               # Mosaic增强
                close_mosaic=10,          # 关闭Mosaic的epoch
                mixup=0.0,                # 混合增强
                erasing=0.1,              # 随机擦除
                translate=0.1,            # 平移增强
                scale=0.5,                # 缩放增强
                
                # === 损失函数权重 ===
                box=7.5,                  # 边界框损失权重
                cls=0.5,                  # 分类损失权重

                # === 验证设置 ===
                val=True,                 # 训练时进行验证
                split='val',              # 验证集分割
                save_json=False,          # 不保存JSON结果
                save_hybrid=False,        # 不保存hybrid标签
            )
            
            LOGGER.info("✅ 训练完成！")
            
            # 从训练结果中获取实际保存目录
            save_dir = Path(model.trainer.save_dir)
            LOGGER.info(f"训练结果保存在: {save_dir}")
            
            # 返回最佳模型路径
            best_model = save_dir / "weights" / "best.pt"
            last_model = save_dir / "weights" / "last.pt"
            
            if best_model.exists():
                LOGGER.info(f"✅ 最佳模型: {best_model}")
                return str(best_model)
            elif last_model.exists():
                LOGGER.info(f"⚠️ 未找到best.pt，返回last.pt: {last_model}")
                return str(last_model)
            else:
                LOGGER.warning(f"❌ 模型文件不存在于: {save_dir / 'weights'}")
                # 尝试在runs/detect下查找最新的训练目录
                runs_dir = Path("runs/detect")
                if runs_dir.exists():
                    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                        LOGGER.info(f"尝试从最新训练目录获取: {latest_dir}")
                        best_model = latest_dir / "weights" / "best.pt"
                        last_model = latest_dir / "weights" / "last.pt"
                        if best_model.exists():
                            return str(best_model)
                        elif last_model.exists():
                            return str(last_model)
            
            return None
            
        except Exception as e:
            LOGGER.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_pipeline(
        self,
        # 切片参数
        crop_size: int = 640,
        gap: int = 100,
        rates: Tuple[float, ...] = (1.0,),
        force_slice: bool = False,
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
        
        # 2. 数据切片
        if not self.slice_data(crop_size, gap, rates, force_slice):
            return False
        
        # 3. 模型训练
        best_model = self.train_model(epochs, imgsz, batch, device, patience, resume)
        if best_model is None:
            return False
        
        LOGGER.info("🎉 完整训练流水线执行成功！")
        LOGGER.info(f"最佳模型: {best_model}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Balloon 数据切片和训练集成脚本 (单尺度版本)")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, 
                       default="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format",
                       help="原始 Balloon 数据根目录")
    parser.add_argument("--slice-dir", type=str, 
                       default="/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format_slice",
                       help="切片后数据保存目录")
    parser.add_argument("--project-name", type=str, default="D1_yolo11l_slice", help="训练项目名称")
    
    # 切片参数
    parser.add_argument("--crop-size", type=int, default=640, help="切片窗口大小")
    parser.add_argument("--gap", type=int, default=100, help="窗口重叠大小")
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0], help="多尺度缩放比例")
    parser.add_argument("--force-slice", action="store_true", help="强制重新切片")
    
    # 训练参数
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="模型名称或路径")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备编号")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    
    # 模式选择
    parser.add_argument("--slice-only", action="store_true", help="仅执行数据切片")
    parser.add_argument("--train-only", action="store_true", help="仅执行模型训练")
    
    args = parser.parse_args()
    
    try:
        # 创建训练流水线
        pipeline = BalloonTrainingPipeline(
            data_root=args.data_root,
            slice_dir=args.slice_dir,
            model_name=args.model,
            project_name=args.project_name
        )
        
        if args.slice_only:
            # 仅执行切片
            success = pipeline.slice_data(
                crop_size=args.crop_size,
                gap=args.gap,
                rates=tuple(args.rates),
                force_slice=args.force_slice
            )
        elif args.train_only:
            # 仅执行训练
            best_model = pipeline.train_model(
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                patience=args.patience,
                resume=args.resume
            )
            success = best_model is not None
        else:
            # 执行完整流水线
            success = pipeline.run_complete_pipeline(
                crop_size=args.crop_size,
                gap=args.gap,
                rates=tuple(args.rates),
                force_slice=args.force_slice,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                patience=args.patience,
                resume=args.resume
            )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        LOGGER.error(f"❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

