#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balloon数据集YOLO11训练脚本
使用YOLO11预训练模型进行迁移学习，训练气球检测器
"""

from ultralytics import YOLO
import os
import argparse
from pathlib import Path


def train_balloon_detector(
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch: int = 8,
    device: int = 5,
    patience: int = 10,
    project_name: str = "balloon_exp"
):
    """训练气球检测器"""
    
    print("🎈 开始训练Balloon检测器")
    print("=" * 60)
    
    # 配置文件路径
    config_file = Path("/home/cjh/ultralytics/my_balloon.yaml").expanduser()
    
    # 检查配置文件是否存在
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        print("请先运行数据集转换脚本: python convert_balloon_to_yolo.py")
        return
    
    print(f"📁 使用配置文件: {config_file}")
    print(f"🔧 训练参数: model={model_name}, epochs={epochs}, batch={batch}, device={device}, patience={patience}")
    
    # 1. 加载YOLO11预训练模型
    print("\n📦 加载YOLO11预训练模型...")
    model = YOLO(model_name)
    
    # 显示模型信息
    print("🔍 模型信息:")
    model.info(verbose=False)
    
    # 2. 开始训练 - 迁移学习
    print("\n🚀 开始训练 (迁移学习)...")
    print("💡 使用预训练权重，只需少量epoch即可获得好效果")
    
    results = model.train(
        # === 数据集配置 ===
        data=str(config_file),    # 数据集配置文件
        
        # === 基础训练参数 ===
        epochs=epochs,            # 训练轮数
        batch=batch,              # 批大小
        imgsz=1024,                # 输入图像尺寸
        device=device,            # GPU设备
        
        # === 项目管理 ===
        project='runs/detect',    # 项目目录
        name=project_name,        # 实验名称
        exist_ok=True,            # 允许覆盖现有实验
        
        # === 迁移学习优化参数 ===
        # 注意：小数据集(61张)需要更保守的学习率
        lr0=0.001,                # 初始学习率 (YOLO默认值，小数据集适用)
        lrf=0.01,                 # 最终学习率比例 (更小，避免后期震荡)
        momentum=0.937,           # 动量
        weight_decay=0.0005,      # 权重衰减
        warmup_epochs=5,          # 预热轮数 (增加，让模型稳定过渡)
        warmup_momentum=0.8,      # 预热动量
        
        # === 早停和保存 ===
        patience=patience,        # 早停耐心值
        save_period=20,           # 每20轮保存一次
        
        # === 数据增强 (小数据集需要更温和的增强，避免过拟合) ===
        hsv_h=0.015,             # 色调增强
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 明度增强
        degrees=0.0,             # 旋转角度 (关闭，小数据集容易导致不稳定)
        translate=0.1,           # 平移比例
        scale=0.5,               # 缩放比例
        shear=0.0,               # 剪切角度 (关闭)
        perspective=0.0,         # 透视变换 (关闭)
        fliplr=0.5,              # 水平翻转概率
        flipud=0.0,              # 垂直翻转概率
        mosaic=1.0,              # 马赛克增强 (全开，小数据集有帮助)
        mixup=0.0,               # MixUp增强 (关闭)
        copy_paste=0.0,          # 复制粘贴增强 (关闭)
        
        # === 训练优化 ===
        amp=True,                # 自动混合精度训练
        cache=True,              # 缓存图像 (数据集小，可以缓存加速)
        rect=False,              # 不使用矩形训练
        cos_lr=True,             # 使用余弦学习率调度 (更平滑的学习率下降)
        close_mosaic=10,         # 最后10轮关闭马赛克增强
        
        # === 其他设置 ===
        workers=4,               # 数据加载线程数
        verbose=True,            # 详细输出
        seed=42,                 # 随机种子，保证可重现性
        deterministic=True,      # 确定性训练
        single_cls=False,        # 多类别训练 (虽然只有1类)
        plots=True,              # 生成训练图表
        
        
        # === 验证设置 ===
        val=True,                # 训练时进行验证
        split='val',             # 验证集分割
        save_json=False,         # 不保存JSON结果 (单类别不需要)
        save_hybrid=False,       # 不保存hybrid标签
        
        # === 冻结层设置 (可选的迁移学习策略) ===
        # freeze=None,           # 不冻结任何层 (推荐)
        # freeze=[0, 1, 2],      # 冻结前几层 (可选)
    )
    
    print("\n✅ 训练完成!")
    print(f"📊 训练结果保存在: runs/detect/{project_name}/")
    print(f"🏆 最佳模型: runs/detect/{project_name}/weights/best.pt")
    print(f"📈 训练曲线: runs/detect/{project_name}/results.png")
    
    return results, project_name


def validate_model(project_name: str = "balloon_exp"):
    """验证训练好的模型"""
    print("\n🔍 验证训练好的模型...")
    
    # 加载最佳模型
    model_path = f"runs/detect/{project_name}/weights/best.pt"
    config_file = Path("/home/cjh/ultralytics/my_balloon.yaml").expanduser()
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先完成训练！")
        return
    
    model = YOLO(model_path)
    
    # 在验证集上评估
    results = model.val(
        data=str(config_file),
        batch=8,
        imgsz=1024,
        conf=0.25,               # 置信度阈值
        iou=0.5,                 # NMS IoU阈值
        save_json=True,          # 保存详细结果
        plots=True,              # 生成验证图表
        verbose=True
    )
    
    print("✅ 验证完成!")
    print(f"📊 验证结果: mAP@0.5 = {results.box.map50:.3f}")
    print(f"📊 验证结果: mAP@0.5:0.95 = {results.box.map:.3f}")
    
    return results


def predict_sample(project_name: str = "balloon_exp"):
    """使用训练好的模型进行预测"""
    print("\n🔮 使用模型进行预测...")
    
    model_path = f"runs/detect/{project_name}/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 在验证集的一张图像上进行预测
    val_images_dir = "/home/cjh/mmdetection/data/balloon/yolo_format/images/val"
    sample_images = list(Path(val_images_dir).glob("*.jpg"))[:3]  # 取前3张图像
    
    if sample_images:
        print(f"📸 对 {len(sample_images)} 张样例图像进行预测...")
        
        results = model.predict(
            source=sample_images,
            conf=0.25,              # 置信度阈值
            iou=0.5,                # NMS IoU阈值
            save=True,              # 保存预测结果
            save_txt=True,          # 保存txt格式结果
            save_conf=True,         # 保存置信度
            show_labels=True,       # 显示标签
            show_conf=True,         # 显示置信度
            line_width=2,           # 边界框线宽
            project='runs/detect',  # 项目目录
            name='balloon_pred',    # 预测结果名称
            exist_ok=True          # 覆盖现有结果
        )
        
        print("✅ 预测完成!")
        print(f"📁 预测结果保存在: runs/detect/balloon_pred/")
    else:
        print("❌ 没有找到验证图像进行预测")


def export_model(project_name: str = "balloon_exp"):
    """导出模型为不同格式"""
    print("\n📤 导出模型...")
    
    model_path = f"runs/detect/{project_name}/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 导出为ONNX格式 (用于生产部署)
    print("🔄 导出ONNX格式...")
    onnx_path = model.export(
        format='onnx',          # 导出格式
        imgsz=1024,              # 输入尺寸
        simplify=True,          # 简化模型
        dynamic=False,          # 固定输入尺寸
        opset=11                # ONNX opset版本
    )
    
    print(f"✅ ONNX模型已导出: {onnx_path}")
    print("💡 ONNX模型可用于C++、Java、移动端等部署")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Balloon数据集YOLO11训练脚本")
    
    # 训练参数
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="模型名称或路径")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--device", type=int, default=5, help="GPU设备编号")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--project-name", type=str, default="balloon_exp", help="项目名称")
    
    # 执行步骤
    parser.add_argument("--train-only", action="store_true", help="仅训练，不进行验证和预测")
    parser.add_argument("--skip-export", action="store_true", help="跳过模型导出")
    
    args = parser.parse_args()
    
    print("🎈 Balloon检测器训练管道")
    print("=" * 60)
    print(f"📋 接收到的参数: model={args.model}, epochs={args.epochs}, batch={args.batch}")
    
    try:
        # 步骤1: 训练模型
        print("第1步: 训练模型")
        train_results, project_name = train_balloon_detector(
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            project_name=args.project_name
        )
        
        if args.train_only:
            print("\n✅ 训练完成！(仅训练模式)")
            return
        
        # 步骤2: 验证模型
        print("\n第2步: 验证模型")
        val_results = validate_model(project_name)
        
        # 步骤3: 样例预测
        print("\n第3步: 样例预测")
        predict_sample(project_name)
        
        # 步骤4: 导出模型
        if not args.skip_export:
            print("\n第4步: 导出模型")
            export_model(project_name)
        
        print("\n" + "=" * 60)
        print("🎉 所有步骤完成!")
        print(f"📊 查看训练结果: runs/detect/{project_name}/results.png")
        print(f"🔮 查看预测结果: runs/detect/balloon_pred/")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("💡 请检查:")
        print("  1. 数据集是否已正确转换")
        print("  2. GPU设备是否可用")
        print("  3. 内存是否足够")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 