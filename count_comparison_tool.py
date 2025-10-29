#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测数量对比工具 - 简化版本
通过比较真实标签和预测标签文件的行数来评估检测数量准确性

使用方法:
1. 修改下面的路径变量
2. 运行脚本: python count_comparison_tool.py

作者: AI Assistant
日期: 2025-01-29
"""

import os
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


def calculate_accuracy(true_dir: str, pred_dir: str):
    """计算检测数量准确性"""
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
    print(f"   平均每文件真实检测数: {total_true/len(results):.2f}")
    print(f"   平均每文件预测检测数: {total_pred/len(results):.2f}")
    
    if valid_results:
        print(f"\n🎯 Metric值统计:")
        print(f"   平均Metric值: {avg_metric:.4f}")
        print(f"   Metric范围: [{min_metric:.4f}, {max_metric:.4f}]")
        
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
    
    print(f"\n🎉 分析完成!")


def main():
    """主函数"""
    print("🔬 检测数量对比工具")
    print("通过比较txt文件行数评估检测数量准确性")
    print("="*60)
    
    # ========== 请修改下面的路径 ==========
    # 真实标签目录路径
    true_labels_dir = "/home/cjh/mmdetection/data/balloon/yolo_format/labels/val"
    
    # 预测标签目录路径  
    pred_labels_dir = "/home/cjh/ultralytics/runs/inference/balloon_yolo11x_20251022_211601_val/predict/labels"
    # =====================================
    
    print(f"📂 真实标签目录: {true_labels_dir}")
    print(f"📂 预测标签目录: {pred_labels_dir}")
    
    # 检查路径是否存在
    if not Path(true_labels_dir).exists():
        print(f"❌ 真实标签目录不存在: {true_labels_dir}")
        return
    
    if not Path(pred_labels_dir).exists():
        print(f"❌ 预测标签目录不存在: {pred_labels_dir}")
        return
    
    # 计算准确性
    calculate_accuracy(true_labels_dir, pred_labels_dir)


if __name__ == "__main__":
    main()