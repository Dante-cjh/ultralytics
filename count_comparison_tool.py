#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测数量对比工具 - 简化版本
通过比较真实标签和预测标签文件的行数来评估检测数量准确性

使用方法:
python count_comparison_tool.py --model_name D1_yolov8l_20251028_174321_val

参数说明:
  --model_name: 模型名称（必需）
  --true_labels_dir: 真实标签目录（可选，有默认值）
  --good_threshold: 好图片阈值（默认0.95）
  --bad_threshold: 坏图片阈值（默认0.1）
  --save_images: 是否保存图片（默认True）

作者: AI Assistant
日期: 2025-01-29
"""

import os
import argparse
import shutil
from pathlib import Path


def count_lines_in_file(file_path: str) -> int:
    """计算txt文件中的非空行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return len([line for line in lines if line.strip()])
    except Exception as e:
        print(f"   ❌ 读取文件 {file_path} 时出错: {e}")
        return 0


def check_file_consistency(true_dir: str, pred_dir: str):
    """检查文件名一致性"""
    print("🔍 检查文件名一致性...")
    print("="*50)
    
    # 获取文件名
    true_files = set(f.stem for f in Path(true_dir).glob("*.txt"))
    pred_files = set(f.stem for f in Path(pred_dir).glob("*.txt"))
    
    print(f"📂 真实标签: {len(true_files)} 个文件")
    print(f"📂 预测标签: {len(pred_files)} 个文件")
    
    # 比较
    common_files = true_files & pred_files
    only_in_true = true_files - pred_files
    only_in_pred = pred_files - true_files
    
    print(f"\n📊 比较结果:")
    print(f"   共同文件: {len(common_files)}")
    print(f"   仅在真实标签中: {len(only_in_true)}")
    print(f"   仅在预测标签中: {len(only_in_pred)}")
    
    if only_in_true:
        print(f"\n⚠️  仅在真实标签中的文件:")
        for filename in sorted(only_in_true)[:5]:
            print(f"   - {filename}.txt")
        if len(only_in_true) > 5:
            print(f"   ... 还有 {len(only_in_true) - 5} 个")
    
    if only_in_pred:
        print(f"\n⚠️  仅在预测标签中的文件:")
        for filename in sorted(only_in_pred)[:5]:
            print(f"   - {filename}.txt")
        if len(only_in_pred) > 5:
            print(f"   ... 还有 {len(only_in_pred) - 5} 个")
    
    if len(common_files) == 0:
        print(f"\n❌ 没有共同文件！请检查路径是否正确。")
        return []
    elif len(only_in_true) == 0 and len(only_in_pred) == 0:
        print(f"\n✅ 文件名完全一致！")
    else:
        print(f"\n⚠️  文件名不完全一致，但可以继续处理共同文件。")
    
    return sorted(common_files)


def calculate_accuracy(true_dir: str, pred_dir: str, save_images: bool = False, 
                      good_threshold: float = 0.95, bad_threshold: float = 0.1,
                      model_name: str = "", images_dir: str = ""):
    """计算检测数量准确性
    
    参数:
        true_dir: 真实标签目录
        pred_dir: 预测标签目录
        save_images: 是否保存图片
        good_threshold: 好图片阈值（准确率高于此值）
        bad_threshold: 坏图片阈值（准确率低于此值）
        model_name: 模型名称（用于构建保存路径）
        images_dir: 图片目录
    """
    print(f"\n📊 计算检测数量准确性...")
    print("="*50)
    
    # 检查文件名一致性
    common_files = check_file_consistency(true_dir, pred_dir)
    
    if not common_files:
        return
    
    # 计算每个文件的检测数量
    results = []
    total_true = 0
    total_pred = 0
    
    print(f"\n🔢 计算 {len(common_files)} 个文件的检测数量...")
    
    for filename in common_files:
        true_file = Path(true_dir) / f"{filename}.txt"
        pred_file = Path(pred_dir) / f"{filename}.txt"
        
        true_count = count_lines_in_file(str(true_file))
        pred_count = count_lines_in_file(str(pred_file))
        
        # 计算metric: 1 - |pred - true| / true
        if true_count > 0:
            metric = 1 - abs(pred_count - true_count) / true_count
        else:
            metric = 1.0 if pred_count == 0 else float('-inf')
        
        results.append({
            "filename": filename,
            "true_count": true_count,
            "pred_count": pred_count,
            "metric": metric,
            "diff": abs(pred_count - true_count)
        })
        
        total_true += true_count
        total_pred += pred_count
    
    # 过滤有效结果
    valid_results = [r for r in results if r["metric"] != float('-inf')]
    
    # 计算统计信息
    if valid_results:
        avg_metric = sum(r["metric"] for r in valid_results) / len(valid_results)
        min_metric = min(r["metric"] for r in valid_results)
        max_metric = max(r["metric"] for r in valid_results)
        
        # 排序找出最好和最差的
        sorted_results = sorted(valid_results, key=lambda x: x["metric"], reverse=True)
        top_5 = sorted_results[:5]
        bottom_5 = sorted_results[-5:]
    else:
        avg_metric = min_metric = max_metric = 0
        top_5 = bottom_5 = []
    
    # 打印结果
    print(f"\n📈 总体统计:")
    print(f"   处理文件数: {len(results)}")
    print(f"   有效文件数: {len(valid_results)}")
    print(f"   总真实检测数: {total_true}")
    print(f"   总预测检测数: {total_pred}")
    print(f"   差值: {total_pred - total_true} ({'+' if total_pred >= total_true else ''}{total_pred - total_true})")
    print(f"   平均每文件真实检测数: {total_true/len(results):.2f}")
    print(f"   平均每文件预测检测数: {total_pred/len(results):.2f}")
    
    # 计算全局Metric（基于总数）
    if total_true > 0:
        global_metric = 1 - abs(total_pred - total_true) / total_true
        global_error_rate = abs(total_pred - total_true) / total_true
    else:
        global_metric = 1.0 if total_pred == 0 else 0.0
        global_error_rate = 0.0
    
    if valid_results:
        print(f"\n🎯 Metric值统计 (两种计算方法):")
        print(f"   ┌─ 方法1: 样本平均Metric = {avg_metric:.4f}")
        print(f"   │  说明: 先计算每个样本的metric，再求平均")
        print(f"   │  计算: mean([metric_1, metric_2, ..., metric_n])")
        print(f"   │")
        print(f"   └─ 方法2: 全局总数Metric = {global_metric:.4f}")
        print(f"      说明: 基于所有预测框总数 vs 所有真实框总数")
        print(f"      计算: 1 - |{total_pred} - {total_true}| / {total_true} = {global_metric:.4f}")
        print(f"      全局误差率: {global_error_rate:.2%}")
        print(f"")
        print(f"   样本Metric范围: [{min_metric:.4f}, {max_metric:.4f}]")
        
        # 分类统计
        perfect = sum(1 for r in valid_results if r["metric"] == 1.0)
        good = sum(1 for r in valid_results if r["metric"] >= 0.8)
        poor = sum(1 for r in valid_results if r["metric"] < 0.5)
        
        print(f"   完美匹配: {perfect} 个 ({perfect/len(valid_results)*100:.1f}%)")
        print(f"   良好预测: {good} 个 ({good/len(valid_results)*100:.1f}%)")
        print(f"   较差预测: {poor} 个 ({poor/len(valid_results)*100:.1f}%)")
        
        print(f"\n🏆 准确度最高的5个文件:")
        for i, result in enumerate(top_5):
            print(f"   {i+1}. {result['filename']}.txt: "
                  f"真实={result['true_count']}, 预测={result['pred_count']}, "
                  f"Metric={result['metric']:.4f}")
        
        print(f"\n⚠️  准确度最低的5个文件:")
        for i, result in enumerate(bottom_5):
            print(f"   {i+1}. {result['filename']}.txt: "
                  f"真实={result['true_count']}, 预测={result['pred_count']}, "
                  f"Metric={result['metric']:.4f}")
    
    # 保存好图片和坏图片
    if save_images and model_name and images_dir:
        print(f"\n💾 开始保存图片...")
        print(f"   好图片阈值: {good_threshold}")
        print(f"   坏图片阈值: {bad_threshold}")
        
        # 创建保存目录
        save_base_dir = Path("/public/home/baichen/download/dcu_yolo/ultralytics/runs/good_bad_imgs") / model_name
        good_img_dir = save_base_dir / "good_img"
        bad_img_dir = save_base_dir / "bad_img"
        
        good_img_dir.mkdir(parents=True, exist_ok=True)
        bad_img_dir.mkdir(parents=True, exist_ok=True)
        
        good_count = 0
        bad_count = 0
        
        # 遍历所有结果，保存符合条件的图片
        for result in valid_results:
            metric = result["metric"]
            filename = result["filename"]
            
            # 查找对应的图片文件（支持常见的图片格式）
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                image_path = Path(images_dir) / f"{filename}{ext}"
                if image_path.exists():
                    image_found = True
                    
                    # 判断是好图片还是坏图片
                    if metric >= good_threshold:
                        # 复制到good_img目录
                        dst_path = good_img_dir / f"{filename}{ext}"
                        shutil.copy2(str(image_path), str(dst_path))
                        good_count += 1
                    elif metric <= bad_threshold:
                        # 复制到bad_img目录
                        dst_path = bad_img_dir / f"{filename}{ext}"
                        shutil.copy2(str(image_path), str(dst_path))
                        bad_count += 1
                    
                    break
            
            if not image_found and (metric >= good_threshold or metric <= bad_threshold):
                print(f"   ⚠️  未找到图片: {filename}")
        
        print(f"\n✅ 图片保存完成!")
        print(f"   保存路径: {save_base_dir}")
        print(f"   好图片数量: {good_count} 个 (准确率 >= {good_threshold})")
        print(f"   坏图片数量: {bad_count} 个 (准确率 <= {bad_threshold})")
    elif save_images:
        print(f"\n⚠️  未能保存图片: 缺少必要参数（model_name或images_dir）")
    
    print(f"\n🎉 分析完成!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='检测数量对比工具 - 评估YOLO检测数量准确性',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称，例如: balloon_yolo11x_20251022_211601_val'
    )
    
    parser.add_argument(
        '--true_labels_dir',
        type=str,
        default='/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format/labels/val',
        help='真实标签目录路径 (默认: /public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format/labels/val)'
    )
    
    parser.add_argument(
        '--good_threshold',
        type=float,
        default=1,
        help='好图片的准确率阈值 (默认: 0.95)'
    )
    
    parser.add_argument(
        '--bad_threshold',
        type=float,
        default=0.3,
        help='坏图片的准确率阈值 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--save_images',
        type=lambda x: x.lower() in ['true', '1', 'yes', 'y'],
        default=True,
        help='是否保存图片 (默认: True)'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='/public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format/images/val',
        help='图片目录路径 (默认: /public/home/baichen/download/dcu_yolo/ultralytics/data/D1_type3/yolo_format/images/val)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    print("🔬 检测数量对比工具")
    print("通过比较txt文件行数评估检测数量准确性")
    print("="*60)
    
    # 构建预测标签目录路径
    pred_labels_dir = f"/public/home/baichen/download/dcu_yolo/ultralytics/runs/inference/{args.model_name}/predict/labels"
    
    print(f"\n📋 参数设置:")
    print(f"   模型名称: {args.model_name}")
    print(f"   真实标签目录: {args.true_labels_dir}")
    print(f"   预测标签目录: {pred_labels_dir}")
    print(f"   图片目录: {args.images_dir}")
    print(f"   好图片阈值: {args.good_threshold}")
    print(f"   坏图片阈值: {args.bad_threshold}")
    print(f"   保存图片: {'是' if args.save_images else '否'}")
    
    # 检查路径是否存在
    if not Path(args.true_labels_dir).exists():
        print(f"\n❌ 真实标签目录不存在: {args.true_labels_dir}")
        return
    
    if not Path(pred_labels_dir).exists():
        print(f"\n❌ 预测标签目录不存在: {pred_labels_dir}")
        return
    
    if args.save_images and not Path(args.images_dir).exists():
        print(f"\n⚠️  警告: 图片目录不存在: {args.images_dir}")
        print(f"   将继续进行准确率计算，但不会保存图片")
        args.save_images = False
    
    # 计算准确性
    calculate_accuracy(
        true_dir=args.true_labels_dir,
        pred_dir=pred_labels_dir,
        save_images=args.save_images,
        good_threshold=args.good_threshold,
        bad_threshold=args.bad_threshold,
        model_name=args.model_name,
        images_dir=args.images_dir
    )


if __name__ == "__main__":
    main()