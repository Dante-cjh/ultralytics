#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOTA到YOLO OBB格式转换脚本
使用ultralytics官方的convert_dota_to_yolo_obb函数进行转换
首先重组数据结构，然后调用官方函数
"""

import os
import shutil
from pathlib import Path
from ultralytics.data.converter import convert_dota_to_yolo_obb as official_convert

def reorganize_dota_data(original_path: str, target_path: str):
    """
    重新组织DOTA数据结构以匹配官方函数期望的格式
    
    Args:
        original_path (str): 原始DOTA数据路径
        target_path (str): 目标路径（重组后的数据路径）
    """
    original_path = Path(original_path)
    target_path = Path(target_path)
    
    print(f"开始重新组织DOTA数据结构...")
    print(f"原始路径：{original_path}")
    print(f"目标路径：{target_path}")
    
    # 创建目标目录结构
    for phase in ["train", "val"]:
        # 创建images目录
        images_target_dir = target_path / "images" / phase
        images_target_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建labels的original目录  
        labels_target_dir = target_path / "labels" / f"{phase}_original"
        labels_target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图像文件
        images_source_dir = original_path / phase / "images" / "images"
        if images_source_dir.exists():
            print(f"复制{phase}图像文件...")
            for img_file in images_source_dir.glob("*.png"):
                target_img_file = images_target_dir / img_file.name
                if not target_img_file.exists():
                    shutil.copy2(img_file, target_img_file)
        
        # 处理标签文件 - 需要去掉前两行元信息
        labels_source_dir = original_path / phase / "labelTxt"
        if labels_source_dir.exists():
            print(f"处理{phase}标签文件...")
            for label_file in labels_source_dir.glob("*.txt"):
                target_label_file = labels_target_dir / label_file.name
                
                # 读取原始标签文件，跳过前两行元信息
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 过滤掉元信息行和空行，只保留标注数据
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith("imagesource:") and 
                        not line.startswith("gsd:")):
                        filtered_lines.append(line)
                
                # 写入目标标签文件
                with open(target_label_file, 'w', encoding='utf-8') as f:
                    for line in filtered_lines:
                        f.write(line + '\n')
    
    print(f"✓ 数据结构重组完成！")
    return target_path

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    使用官方函数将DOTA数据集标注转换为YOLO OBB格式
    
    Args:
        dota_root_path (str): DOTA数据集根目录路径
    """
    print(f"开始使用官方函数进行DOTA到YOLO OBB格式转换...")
    print(f"数据根目录：{dota_root_path}")
    
    # 检查数据集路径是否存在
    dota_path = Path(dota_root_path)
    if not dota_path.exists():
        print(f"错误：数据集路径不存在 - {dota_path}")
        return False
    
    try:
        # 调用官方转换函数
        official_convert(str(dota_path))
        print(f"✓ 官方转换函数执行完成！")
        return True
    except Exception as e:
        print(f"错误：转换过程中出现问题 - {e}")
        return False

if __name__ == "__main__":
    print("DOTA到YOLO OBB格式转换工具")
    print("=" * 50)
    
    # 原始数据路径
    original_dota_path = "/home/cjh/mmdetection/data/dota"
    # 重组后的数据路径
    target_dota_path = "/home/cjh/ultralytics/dota_reorganized"
    
    try:
        # 步骤1：重新组织数据结构
        print("步骤1：重新组织数据结构以匹配官方函数期望...")
        reorganize_dota_data(original_dota_path, target_dota_path)
        
        # 步骤2：使用官方函数进行转换
        print("\n步骤2：使用官方函数进行DOTA到YOLO OBB转换...")
        success = convert_dota_to_yolo_obb(target_dota_path)
        
        if success:
            print("\n✅ 转换完成！数据集已准备就绪，可以开始训练。")
            print(f"转换后的数据集位置：{target_dota_path}")
            print("\n最终目录结构：")
            for phase in ["train", "val"]:
                images_dir = Path(target_dota_path) / "images" / phase
                labels_dir = Path(target_dota_path) / "labels" / phase
                if images_dir.exists() and labels_dir.exists():
                    img_count = len(list(images_dir.glob("*.png")))
                    label_count = len(list(labels_dir.glob("*.txt")))
                    print(f"  {phase}: {img_count} 个图像文件, {label_count} 个标签文件")
        else:
            print("\n❌ 转换过程中出现错误")
            
    except Exception as e:
        print(f"\n❌ 转换过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
