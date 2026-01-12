#!/usr/bin/env python3
"""
COCO数据集转换为YOLO格式的脚本
"""

import json
import os
import shutil
from tqdm import tqdm

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """将COCO格式的边界框转换为YOLO格式"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return [x_center, y_center, w, h]

def convert_coco_to_yolo(coco_annotations_path, output_dir, dataset_type):
    """转换COCO标注文件为YOLO格式"""
    
    # 创建输出目录
    labels_dir = os.path.join(output_dir, 'labels', dataset_type)
    images_dir = os.path.join(output_dir, 'images', dataset_type)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 读取COCO标注文件
    print(f"正在读取 {coco_annotations_path}...")
    with open(coco_annotations_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建图像ID到图像信息的映射
    images_info = {}
    for img in coco_data['images']:
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # 创建类别ID到索引的映射
    categories = coco_data['categories']
    category_id_to_index = {}
    for i, cat in enumerate(categories):
        category_id_to_index[cat['id']] = i
    
    print(f"类别映射:")
    for cat in categories:
        print(f"  {cat['id']} -> {category_id_to_index[cat['id']]} ({cat['name']})")
    
    # 按图像ID组织标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 处理每张图像
    print(f"正在转换 {dataset_type} 数据集...")
    processed_count = 0
    
    for img_id, annotations in tqdm(annotations_by_image.items(), desc=f"转换 {dataset_type}"):
        if img_id not in images_info:
            continue
            
        img_info = images_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 创建YOLO格式的标注文件
        label_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                category_index = category_id_to_index[ann['category_id']]
                bbox = ann['bbox']
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                line = f"{category_index} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n"
                f.write(line)
        
        # 复制图像文件
        src_image_path = os.path.join('data/coco', f'{dataset_type}2017', img_info['file_name'])
        dst_image_path = os.path.join(images_dir, img_info['file_name'])
        
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
            processed_count += 1
        else:
            print(f"警告: 图像文件不存在 {src_image_path}")
    
    print(f"{dataset_type} 数据集转换完成，处理了 {processed_count} 张图像")
    return processed_count

def create_dataset_yaml(output_dir, categories):
    """创建数据集配置文件"""
    yaml_content = f"""# YOLO数据集配置文件
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
nc: {len(categories)}
names:
"""
    
    for i, cat in enumerate(categories):
        yaml_content += f"  {i}: {cat['name']}\n"
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_path}")

def main():
    # 检查输入文件
    train_annotations = './data/coco/annotations/instances_train2017.json'
    val_annotations = './data/coco/annotations/instances_val2017.json'
    output_dir = './data/coco/yolo_format'
    
    if not os.path.exists(train_annotations):
        print(f"错误: 训练集标注文件不存在 {train_annotations}")
        return
    
    if not os.path.exists(val_annotations):
        print(f"错误: 验证集标注文件不存在 {val_annotations}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取类别信息
    with open(train_annotations, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    categories = train_data['categories']
    
    # 转换训练集和验证集
    train_count = convert_coco_to_yolo(train_annotations, output_dir, 'train')
    val_count = convert_coco_to_yolo(val_annotations, output_dir, 'val')
    
    # 创建数据集配置文件
    create_dataset_yaml(output_dir, categories)
    
    print(f"\n转换完成!")
    print(f"训练集: {train_count} 张图像")
    print(f"验证集: {val_count} 张图像")
    print(f"类别数: {len(categories)}")
    print(f"输出目录: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main()
