#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOTA 数据切片工具
基于 ultralytics 内置的数据切片功能，提供简单易用的接口
支持多尺度切片、自定义窗口大小、重叠度等参数
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

from ultralytics.data.split_dota import split_trainval, split_test
from ultralytics.utils import LOGGER


def setup_logger():
    """设置日志输出格式"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def check_data_structure(data_root: Path) -> bool:
    """
    检查 DOTA 数据集的目录结构是否正确
    
    Args:
        data_root (Path): 数据根目录
    
    Returns:
        bool: 结构是否正确
    """
    required_dirs = [
        "images/train",
        "images/val", 
        "labels/train",
        "labels/val"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if not full_path.exists():
            missing_dirs.append(str(full_path))
    
    if missing_dirs:
        LOGGER.error(f"缺少以下目录:")
        for missing_dir in missing_dirs:
            LOGGER.error(f"  - {missing_dir}")
        LOGGER.error(f"期望的目录结构:")
        LOGGER.error(f"  {data_root}/")
        LOGGER.error(f"  ├── images/")
        LOGGER.error(f"  │   ├── train/")
        LOGGER.error(f"  │   └── val/")
        LOGGER.error(f"  └── labels/")
        LOGGER.error(f"      ├── train/")
        LOGGER.error(f"      └── val/")
        return False
    
    return True


def get_dataset_info(data_root: Path) -> dict:
    """
    获取数据集基本信息
    
    Args:
        data_root (Path): 数据根目录
    
    Returns:
        dict: 数据集信息
    """
    info = {}
    for split in ["train", "val"]:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        
        img_files = list(img_dir.glob("*.*")) if img_dir.exists() else []
        lbl_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        
        info[split] = {
            "images": len(img_files),
            "labels": len(lbl_files)
        }
    
    return info


def slice_dota_dataset(
    data_root: str,
    save_dir: str,
    crop_size: int = 1024,
    gap: int = 200,
    rates: Tuple[float, ...] = (1.0,),
    include_test: bool = False
):
    """
    切片 DOTA 数据集
    
    Args:
        data_root (str): 原始数据根目录
        save_dir (str): 保存切片后数据的目录
        crop_size (int): 基础裁剪尺寸
        gap (int): 窗口间重叠大小
        rates (Tuple[float, ...]): 多尺度缩放比例
        include_test (bool): 是否包含测试集
    """
    data_root = Path(data_root)
    save_dir = Path(save_dir)
    
    LOGGER.info("🚀 开始 DOTA 数据集切片处理")
    LOGGER.info(f"原始数据路径: {data_root}")
    LOGGER.info(f"输出路径: {save_dir}")
    LOGGER.info(f"切片参数:")
    LOGGER.info(f"  - 基础裁剪尺寸: {crop_size}x{crop_size}")
    LOGGER.info(f"  - 重叠大小: {gap}")
    LOGGER.info(f"  - 多尺度比例: {rates}")
    
    # 检查数据结构
    if not check_data_structure(data_root):
        return False
    
    # 获取数据集信息
    dataset_info = get_dataset_info(data_root)
    LOGGER.info(f"数据集信息:")
    for split, info in dataset_info.items():
        LOGGER.info(f"  - {split}: {info['images']} 图像, {info['labels']} 标签")
    
    # 创建输出目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 切片训练和验证集
        LOGGER.info("📸 开始切片训练和验证集...")
        split_trainval(
            data_root=str(data_root),
            save_dir=str(save_dir),
            crop_size=crop_size,
            gap=gap,
            rates=rates
        )
        LOGGER.info("✅ 训练和验证集切片完成")
        
        # 如果需要，切片测试集
        if include_test:
            test_dir = data_root / "images" / "test"
            if test_dir.exists():
                LOGGER.info("📸 开始切片测试集...")
                split_test(
                    data_root=str(data_root),
                    save_dir=str(save_dir),
                    crop_size=crop_size,
                    gap=gap,
                    rates=rates
                )
                LOGGER.info("✅ 测试集切片完成")
            else:
                LOGGER.warning(f"未找到测试集目录: {test_dir}")
        
        # 获取切片后的数据集信息
        LOGGER.info("📊 切片后数据集统计:")
        sliced_info = get_dataset_info(save_dir)
        for split, info in sliced_info.items():
            LOGGER.info(f"  - {split}: {info['images']} 图像, {info['labels']} 标签")
        
        LOGGER.info(f"🎉 数据切片完成！输出路径: {save_dir}")
        return True
        
    except Exception as e:
        LOGGER.error(f"❌ 切片过程中出现错误: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DOTA 数据集切片工具")
    
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="原始 DOTA 数据集根目录"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="保存切片后数据的目录"
    )
    
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="基础裁剪尺寸 (默认: 1024)"
    )
    
    parser.add_argument(
        "--gap",
        type=int,
        default=200,
        help="窗口间重叠大小 (默认: 200)"
    )
    
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[1.0],
        help="多尺度缩放比例 (默认: [1.0])"
    )
    
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="是否包含测试集切片"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    
    # 执行切片
    success = slice_dota_dataset(
        data_root=args.data_root,
        save_dir=args.save_dir,
        crop_size=args.crop_size,
        gap=args.gap,
        rates=tuple(args.rates),
        include_test=args.include_test
    )
    
    if success:
        print("\n" + "="*50)
        print("🎯 接下来可以使用切片后的数据进行训练:")
        print(f"python -m ultralytics.models.yolo.obb.train data={args.save_dir}")
        print("="*50)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
