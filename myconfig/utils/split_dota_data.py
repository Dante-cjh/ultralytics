#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOTA数据集拆分脚本 - 第一步
功能：按3:1比例拆分DOTA数据集并重组目录结构
"""

import os
import shutil
import random
from pathlib import Path

def split_and_organize_dota_data():
    """按3:1比例拆分DOTA数据并按convert_dota_to_yolo_obb要求组织目录结构"""
    
    # 路径配置
    source_root = Path("/Users/jiahanchen/Desktop/ObjectDetection/Dataset/DOTA")
    source_images_dir = source_root / "val/images/images"
    source_labels_dir = source_root / "val/labelTxt-v1.5/DOTA-v1.5_val"
    
    # 目标路径
    target_root = Path("/Users/jiahanchen/Desktop/ObjectDetection/ultralytics/data/dota")
    
    print("开始处理DOTA数据集拆分...")
    
    # 检查源数据是否存在
    if not source_images_dir.exists():
        print(f"错误：图像目录不存在 - {source_images_dir}")
        return False
    
    if not source_labels_dir.exists():
        print(f"错误：标签目录不存在 - {source_labels_dir}")
        return False
    
    # 获取所有文件名（不包含扩展名）
    image_files = [f.stem for f in source_images_dir.glob("*.png")]
    label_files = [f.stem for f in source_labels_dir.glob("*.txt")]
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 确保图像和标签文件一一对应
    common_files = list(set(image_files) & set(label_files))
    print(f"找到 {len(common_files)} 个配对的图像-标签文件")
    
    if len(common_files) == 0:
        print("错误：未找到配对的图像和标签文件")
        return False
    
    # 随机打乱文件列表
    random.seed(42)  # 设置随机种子以便复现
    random.shuffle(common_files)
    
    # 按3:1比例拆分（75%训练，25%验证）
    split_idx = int(len(common_files) * 0.75)
    train_files = common_files[:split_idx]
    val_files = common_files[split_idx:]
    
    print(f"\n数据拆分结果：")
    print(f"  训练集：{len(train_files)} 个文件")
    print(f"  验证集：{len(val_files)} 个文件")
    
    # 创建目标目录结构
    target_dirs = {
        'images_train': target_root / "images" / "train",
        'images_val': target_root / "images" / "val", 
        'labels_train_original': target_root / "labels" / "train_original",
        'labels_val_original': target_root / "labels" / "val_original"
    }
    
    print("\n创建目标目录结构...")
    # 清理旧目录（如果存在）
    if target_root.exists():
        print(f"清理旧目录：{target_root}")
        shutil.rmtree(target_root)
    
    # 创建新目录
    for dir_name, dir_path in target_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_name}: {dir_path}")
    
    # 复制训练集文件
    print(f"\n复制训练集文件 ({len(train_files)} 个)...")
    copy_files(train_files, source_images_dir, source_labels_dir, 
              target_dirs['images_train'], target_dirs['labels_train_original'])
    
    # 复制验证集文件
    print(f"复制验证集文件 ({len(val_files)} 个)...")
    copy_files(val_files, source_images_dir, source_labels_dir,
              target_dirs['images_val'], target_dirs['labels_val_original'])
    
    print(f"\n✅ 数据重组完成！")
    print(f"目标路径：{target_root}")
    
    # 显示最终的目录结构
    print("\n目录结构：")
    print_directory_structure(target_root)
    
    return True

def copy_files(file_list, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    """复制指定的图像和标签文件到目标目录"""
    for i, filename in enumerate(file_list, 1):
        # 复制图像文件
        src_image = src_images_dir / f"{filename}.png"
        dst_image = dst_images_dir / f"{filename}.png"
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            print(f"警告：图像文件不存在 - {src_image}")
            continue
        
        # 复制标签文件
        src_label = src_labels_dir / f"{filename}.txt"
        dst_label = dst_labels_dir / f"{filename}.txt"
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            print(f"警告：标签文件不存在 - {src_label}")
            continue
            
        if i % 50 == 0:  # 每50个文件报告一次进度
            print(f"  已处理 {i}/{len(file_list)} 个文件...")
    
    print(f"  ✓ 已复制 {len(file_list)} 个文件对")

def print_directory_structure(root_path, max_depth=3, current_depth=0):
    """打印目录结构"""
    if current_depth >= max_depth or not root_path.exists():
        return
        
    indent = "  " * current_depth
    if root_path.is_dir():
        # 获取子目录和文件
        subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        files = [f for f in root_path.iterdir() if f.is_file()]
        
        print(f"{indent}{root_path.name}/")
        
        # 递归打印子目录
        for subdir in subdirs:
            print_directory_structure(subdir, max_depth, current_depth + 1)
        
        # 显示文件数量
        if files and current_depth < max_depth - 1:
            file_count = len(files)
            print(f"{indent}  [{file_count} 个文件]")

if __name__ == "__main__":
    print("DOTA数据集拆分工具")
    print("=" * 50)
    
    success = split_and_organize_dota_data()
    
    if success:
        print("\n🎉 数据拆分和重组完成！")
        print("接下来运行以下命令进行YOLO OBB格式转换：")
        print("python -c \"from ultralytics.data.converter import convert_dota_to_yolo_obb; convert_dota_to_yolo_obb('data/dota')\"")
    else:
        print("\n❌ 处理过程中出现错误，请检查日志信息。")
