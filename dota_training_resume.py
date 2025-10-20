from ultralytics import YOLO

def main():
    print("🚀 继续训练 - 从已训练模型开始新训练...")
    # 使用之前训练好的best.pt作为起点，开始新训练
    model = YOLO('runs/obb/dota_yolo11l_exp2/weights/best.pt')  # 使用最佳模型作为起点

    results = model.train(
        # === 数据集配置 ===
        data='dota.yaml',         # DOTA数据集配置
        
        # === 基础训练参数 ===
        epochs=100,               # 继续训练100个epoch（可根据需要调整）
        imgsz=1280,               # 图像尺寸 (提升小目标检测)
        batch=4,                  # 批大小 (适配4090/24G在1280输入)
        device=5,             # GPU设备

        # === 项目管理 ===
        project='runs/obb',       # 项目目录
        name='dota_yolo11l_continue',  # 新实验名称
        exist_ok=True,            # 允许覆盖现有实验

        # === 早停和保存 ===
        patience=20,              # 早停耐心值
        save_period=20,           # 每20轮保存一次

        # === 训练优化 ===
        amp=True,                # 启用AMP
        cache=False,             # 不缓存图像
        rect=False,              # 不使用矩形训练
        optimizer='AdamW',      # 显式指定，确保 lr0 生效
        lr0=2.2e-4,
        cos_lr=True,
        lrf=0.1,
        warmup_epochs=0.0,

        # === 其他设置 ===
        workers=4,               # 数据加载线程
        verbose=True,            # 详细输出
        seed=42,                 # 随机种子
        deterministic=True,      # 确定性训练
        single_cls=False,        # 多类别训练
        plots=True,              # 生成训练图表

        # === 数据增强设置 ===
        degrees=180,             # 旋转等变性
        flipud=0.5,              # 竖直翻转
        fliplr=0.5,              # 水平翻转
        mosaic=1.0,              # Mosaic增强
        close_mosaic=10,         # 关闭Mosaic的epoch
        mixup=0.1,               # 轻量混合增强
        erasing=0.2,             # 随机擦除
        translate=0.2,           # 平移增强
        
        # === 损失函数权重 ===
        box=9.0,                 # 适度提高定位损失权重

        # === 验证设置 ===
        val=True,                # 训练时进行验证
        split='val',             # 验证集分割
        save_json=False,         # 不保存JSON结果
        save_hybrid=False,       # 不保存hybrid标签
    )
    print("✅ 继续训练完成!")
    return results

if __name__ == "__main__":
    main()