#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balloon数据集COCO格式转YOLO格式转换脚本
将COCO JSON标注转换为YOLO txt格式，并生成训练所需的文件列表
"""

import json
import os
from pathlib import Path
import shutil


def convert_coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    将COCO格式的bbox转换为YOLO格式
    COCO格式: [x, y, width, height] (左上角坐标)
    YOLO格式: [x_center, y_center, width, height] (归一化，中心坐标)
    """
    x, y, w, h = bbox
    
    # 转换为中心坐标
    x_center = x + w / 2
    y_center = y + h / 2
    
    # 归一化到0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center_norm, y_center_norm, w_norm, h_norm]


def convert_balloon_dataset():
    """转换balloon数据集"""
    
    # 数据集路径
    balloon_root = Path("/home/cjh/mmdetection/data/balloon")
    output_root = balloon_root / "yolo_format"
    
    print(f"🎈 开始转换Balloon数据集")
    print(f"输入路径: {balloon_root}")
    print(f"输出路径: {output_root}")
    
    # 创建输出目录结构
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # 处理训练集和验证集
    for split in ["train", "val"]:
        print(f"\n📂 处理 {split} 数据集...")
        
        # 读取COCO格式的JSON文件
        json_file = balloon_root / split / "annotation_coco.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}
        
        # 创建图像ID到标注的映射
        image_annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # 存储图像路径列表
        image_paths = []
        
        # 处理每张图像
        processed_images = 0
        processed_annotations = 0
        
        for img_id, img_info in image_id_to_info.items():
            filename = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            # 复制图像文件
            src_img_path = balloon_root / split / filename
            dst_img_path = output_root / "images" / split / filename
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
                # 使用绝对路径
                image_paths.append(str(dst_img_path.absolute()))
                processed_images += 1
                
                # 创建对应的标签文件
                label_filename = Path(filename).stem + ".txt"
                label_path = output_root / "labels" / split / label_filename
                
                # 处理该图像的所有标注
                yolo_annotations = []
                if img_id in image_annotations:
                    for ann in image_annotations[img_id]:
                        # 类别ID (balloon是类别0)
                        class_id = ann["category_id"]  # 在balloon数据集中是0
                        
                        # 转换bbox格式
                        bbox = ann["bbox"]
                        yolo_bbox = convert_coco_to_yolo_bbox(bbox, img_width, img_height)
                        
                        # YOLO格式: class_id x_center y_center width height
                        yolo_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                        yolo_annotations.append(yolo_line)
                        processed_annotations += 1
                
                # 写入标签文件
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                    if yolo_annotations:  # 如果有标注，最后加个换行
                        f.write('\n')
            else:
                print(f"⚠️ 图像文件不存在: {src_img_path}")
        
        # 创建图像列表文件
        list_file = output_root / f"{split}.txt"
        with open(list_file, 'w') as f:
            for img_path in sorted(image_paths):
                f.write(f"{img_path}\n")
        
        print(f"✅ {split} 完成: {processed_images} 张图像, {processed_annotations} 个标注")
    
    print(f"\n🎉 转换完成!")
    print(f"📁 YOLO格式数据保存在: {output_root}")
    print(f"📂 目录结构:")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({len(list((output_root / 'images' / 'train').glob('*.jpg')))} 张图像)")
    print(f"  │   └── val/ ({len(list((output_root / 'images' / 'val').glob('*.jpg')))} 张图像)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/ ({len(list((output_root / 'labels' / 'train').glob('*.txt')))} 个标签)")
    print(f"  │   └── val/ ({len(list((output_root / 'labels' / 'val').glob('*.txt')))} 个标签)")
    print(f"  ├── train.txt")
    print(f"  └── val.txt")
    
    return output_root


def verify_conversion():
    """验证转换结果"""
    print(f"\n🔍 验证转换结果...")
    
    output_root = Path("/home/cjh/mmdetection/data/balloon/yolo_format")
    
    # 检查一个标签文件的内容
    train_labels = list((output_root / "labels" / "train").glob("*.txt"))
    if train_labels:
        sample_label = train_labels[0]
        print(f"📄 样例标签文件 {sample_label.name}:")
        with open(sample_label, 'r') as f:
            content = f.read().strip()
            if content:
                lines = content.split('\n')
                for i, line in enumerate(lines[:3], 1):
                    print(f"  {i}: {line}")
                if len(lines) > 3:
                    print(f"  ... (共 {len(lines)} 行)")
            else:
                print("  (空文件 - 该图像无标注)")
    
    # 检查图像列表文件
    train_list = output_root / "train.txt"
    if train_list.exists():
        with open(train_list, 'r') as f:
            lines = f.readlines()
        print(f"📋 train.txt: {len(lines)} 张图像")
        print(f"  前3行: {[line.strip() for line in lines[:3]]}")
    
    print("✅ 验证完成!")


if __name__ == "__main__":
    # 转换数据集
    output_path = convert_balloon_dataset()
    
    # 验证转换结果
    verify_conversion()
    
    print(f"\n💡 接下来请运行:")
    print(f"   cd ~/myconfig")
    print(f"   python balloon_training.py") 