#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用官方Ultralytics API进行COCO训练的简洁脚本
这就是您需要的全部代码！
"""

from ultralytics import YOLO

def main():
    """使用官方API的简洁训练"""
    
    print("🚀 使用官方Ultralytics API训练YOLO模型")
    
    # 1. 加载预训练模型
    model = YOLO('yolo11n.pt')  # 可选: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    
    # 2. 训练模型 - 就这么简单！
    # 注意: train() 会自动使用yaml文件中的val集进行验证
    results = model.train(
        data='my_coco.yaml',      # 数据集配置文件 (包含train和val路径)
        epochs=100,               # 训练轮数
        batch=16,                 # 批大小
        imgsz=640,                # 图像尺寸
        device=5,                 # GPU设备
        project='runs/detect',    # 项目目录
        name='yolo1_balloon_exp',          # 实验名称
        val=True,                 # 训练时自动验证 (默认True)
        
        # 所有训练参数都可以直接传入！
        lr0=0.001,                 # 学习率
        momentum=0.937,           # 动量
        weight_decay=0.0005,      # 权重衰减
        patience=10,              # 早停耐心值
        save_period=10,           # 保存间隔
        
        # 数据增强参数
        hsv_h=0.015,             # 色调增强
        hsv_s=0.7,               # 饱和度增强  
        hsv_v=0.4,               # 明度增强
        degrees=0.0,             # 旋转角度
        translate=0.1,           # 平移
        scale=0.5,               # 缩放
        fliplr=0.5,              # 水平翻转
        mosaic=1.0,              # 马赛克增强
        
        # 其他常用参数
        amp=True,                # 混合精度
        cache=False,             # 图像缓存
        rect=False,              # 矩形训练
        cos_lr=False,            # 余弦学习率
        workers=8,               # 数据加载线程
        verbose=True             # 详细输出
    )
    
    print("✅ 训练完成!")
    print("📊 注意: 训练过程中已自动进行验证，结果保存在 runs/detect/coco_exp/")
    
    # 3. 预测图像 (使用训练好的最佳模型)
    print("🔮 开始预测...")
    pred_results = model.predict(
        source='/home/cjh/mmdetection/data/coco/val2017/000000000139.jpg',
        conf=0.25,
        iou=0.7,
        save=True,
        show_labels=True,
        show_conf=True
    )
    
    print("✅ 预测完成!")
    
    # 4. 导出模型 (可选 - 用于生产部署)
    print("📤 导出模型为ONNX格式 (用于跨平台部署)...")
    print("💡 说明: .pt文件已自动保存在 runs/detect/coco_exp/weights/")
    print("💡 ONNX格式可用于C++、Java、移动端等非Python环境")
    
    export_path = model.export(
        format='onnx',           # 导出格式: onnx, torchscript, tensorflow, etc.
        imgsz=640,
        simplify=True
    )
    
    print(f"✅ ONNX模型已导出: {export_path}")
    
    return results

if __name__ == "__main__":
    main() 