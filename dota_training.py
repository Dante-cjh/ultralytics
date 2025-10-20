#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOTA OBB训练 - 简洁版
使用OBB专用模型进行旋转边界框检测
"""

from ultralytics import YOLO

def main():
    """DOTA OBB训练 - 保持简洁"""
    
    print("🚀 DOTA OBB训练开始...")
    print("📝 使用OBB专用模型: yolo11l-obb.pt")
    
    # 1. 加载OBB专用模型
    model = YOLO('yolo11l-obb.pt')  # 使用OBB专用预训练模型
    
    # 2. 开始训练 - 简化配置
    results = model.train(
        # === 数据集配置 ===
        data='dota.yaml',         # DOTA数据集配置
        
        # === 基础训练参数 ===
        epochs=200,               # 训练轮数
        imgsz=1280,               # 图像尺寸 (提升小目标检测)
        batch=4,                  # 批大小 (适配4090/24G在1280输入)
        device=4,                  # GPU设备

        # === 项目管理 ===
        project='runs/obb',    # 项目目录
        name='dota_yolo11l_exp',       # 实验名称
        exist_ok=False,            # 允许覆盖现有实验

        # === 早停和保存 ===
        patience=30,              # 早停耐心值
        save_period=20,           # 每20轮保存一次

        # === 训练优化 ===
        amp=True,                # 启用AMP (OBB任务兼容性问题)
        cache=False,             # 不缓存图像 (数据集较小)
        rect=False,              # 不使用矩形训练
        cos_lr=True,             # 使用余弦学习率调度 (泛化更稳)
        lr0=0.01,                # 初始学习率
        lrf=0.01,                # 最终学习率因子

        # === 其他设置 ===
        workers=4,               # 单线程数据加载（避免多进程问题）
        verbose=True,            # 详细输出
        seed=42,                 # 随机种子，保证可重现性
        deterministic=True,      # 确定性训练
        single_cls=False,        # 多类别训练 (虽然只有1类)
        plots=True,              # 生成训练图表

        # === 数据增强设置 ===
        degrees=180,             # 旋转等变性 (对遥感/OBB很关键)
        flipud=0.5,              # 竖直翻转 (俯视图收益明显)
        fliplr=0.5,              # 水平翻转
        mosaic=1.0,              # Mosaic增强 (对小目标友好)
        close_mosaic=10,         # 关闭Mosaic的epoch
        mixup=0.1,               # 轻量混合增强
        erasing=0.2,             # 随机擦除 (避免把小目标抹掉)
        translate=0.2,           # 平移增强
        
        # === 损失函数权重 ===
        box=9.0,                 # 适度提高定位损失权重

        # === 验证设置 ===
        val=True,                # 训练时进行验证
        split='val',             # 验证集分割
        save_json=False,         # 不保存JSON结果 (单类别不需要)
        save_hybrid=False,       # 不保存hybrid标签

        # === 冻结层设置 (可选的迁移学习策略) ===
        # freeze=None,           # 不冻结任何层 (推荐)
        # freeze=[0, 1, 2],      # 冻结前几层 (可选)
    )
    
    print("✅ DOTA OBB训练完成!")
    print(f"🏆 最佳模型: runs/obb/train/weights/best.pt")
    
    return results

if __name__ == "__main__":
    main()
