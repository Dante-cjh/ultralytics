#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOTA到YOLO OBB格式转换脚本
功能：将DOTA标签格式转换为YOLO OBB (Oriented Bounding Box) 格式
避免复杂的依赖问题，独立实现转换逻辑
"""

from pathlib import Path
from PIL import Image

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    将DOTA数据集标注转换为YOLO OBB格式
    
    Args:
        dota_root_path (str): DOTA数据集根目录路径
    """
    dota_root_path = Path(dota_root_path)
    
    # DOTA v1.5类别名称到索引的映射（16个类别）
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
    }
    
    print(f"开始DOTA到YOLO OBB格式转换...")
    print(f"数据根目录：{dota_root_path}")
    
    def convert_label(image_name: str, image_width: int, image_height: int, orig_label_dir: Path, save_dir: Path):
        """转换单个图像的DOTA标注到YOLO OBB格式并保存到指定目录"""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"
        
        if not orig_label_path.exists():
            print(f"警告：标签文件不存在 - {orig_label_path}")
            return False
        
        converted_lines = []
        with orig_label_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split()
                if len(parts) < 9:
                    print(f"警告：{orig_label_path} 第{line_num}行格式不正确，跳过")
                    continue
                
                try:
                    # DOTA格式：x1 y1 x2 y2 x3 y3 x4 y4 category difficult
                    class_name = parts[8]
                    if class_name not in class_mapping:
                        print(f"警告：未知类别 '{class_name}' 在 {orig_label_path} 第{line_num}行，跳过")
                        continue
                    
                    class_idx = class_mapping[class_name]
                    coords = [float(p) for p in parts[:8]]
                    
                    # 归一化坐标 (x坐标除以宽度，y坐标除以高度)
                    normalized_coords = [
                        coords[i] / image_width if i % 2 == 0 else coords[i] / image_height 
                        for i in range(8)
                    ]
                    
                    # 格式化坐标（保留6位有效数字）
                    formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                    converted_line = f"{class_idx} {' '.join(formatted_coords)}"
                    converted_lines.append(converted_line)
                    
                except (ValueError, IndexError) as e:
                    print(f"错误：解析 {orig_label_path} 第{line_num}行时出错 - {e}")
                    continue
        
        # 写入转换后的标签文件
        with save_path.open("w", encoding="utf-8") as g:
            for line in converted_lines:
                g.write(line + "\n")
        
        return True
    
    # 处理train和val两个阶段
    total_converted = 0
    for phase in ["train", "val"]:
        print(f"\n处理 {phase} 数据集...")
        
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase
        
        # 检查目录是否存在
        if not image_dir.exists():
            print(f"警告：图像目录不存在 - {image_dir}")
            continue
        
        if not orig_label_dir.exists():
            print(f"警告：原始标签目录不存在 - {orig_label_dir}")
            continue
        
        # 创建输出目录
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录：{save_dir}")
        
        # 获取所有图像文件
        image_paths = list(image_dir.glob("*.png"))
        if not image_paths:
            print(f"警告：在 {image_dir} 中未找到PNG图像文件")
            continue
        
        print(f"找到 {len(image_paths)} 个图像文件")
        converted_count = 0
        
        # 处理每个图像文件
        for i, image_path in enumerate(image_paths, 1):
            image_name_without_ext = image_path.stem
            
            try:
                # 使用PIL获取图像尺寸
                with Image.open(image_path) as img:
                    w, h = img.size
                
                # 转换标签
                success = convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)
                if success:
                    converted_count += 1
                
                # 进度报告
                if i % 50 == 0 or i == len(image_paths):
                    print(f"  已处理 {i}/{len(image_paths)} 个文件...")
                    
            except Exception as e:
                print(f"错误：处理图像 {image_path} 时出错 - {e}")
                continue
        
        print(f"✓ {phase} 数据集转换完成：{converted_count}/{len(image_paths)} 个文件转换成功")
        total_converted += converted_count
    
    print(f"\n🎉 DOTA到YOLO OBB格式转换完成！")
    print(f"总共转换了 {total_converted} 个标签文件")
    
    # 显示最终目录结构
    print(f"\n最终目录结构：")
    for phase in ["train", "val"]:
        label_dir = dota_root_path / "labels" / phase
        if label_dir.exists():
            file_count = len(list(label_dir.glob("*.txt")))
            print(f"  {label_dir}: {file_count} 个标签文件")

if __name__ == "__main__":
    print("DOTA到YOLO OBB格式转换工具")
    print("=" * 50)
    
    # 使用相对路径
    dota_path = "data/dota"
    
    try:
        convert_dota_to_yolo_obb(dota_path)
        print("\n✅ 转换完成！数据集已准备就绪，可以开始训练。")
    except Exception as e:
        print(f"\n❌ 转换过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
