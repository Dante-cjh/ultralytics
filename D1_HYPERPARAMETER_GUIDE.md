# D1_training.py 超参数修改指南

## 📋 修改清单

以下是需要在 `D1_training.py` 中直接修改的参数，按行号列出：

---

### 🔴 必须修改的参数（关键改进）

| 行号 | 原始代码 | 修改为 | 说明 |
|------|----------|--------|------|
| **68** | `lr0=0.0025,` | `lr0=0.001,` | 降低初始学习率，避免过拟合 |
| **72** | `warmup_epochs=3,` | `warmup_epochs=5,` | 增加warmup轮数 |
| **76** | `patience=patience,` | `patience=50,` 或保持参数传入但shell脚本改为50 | 增加早停耐心值 |
| **90** | `mosaic=0.8,` | `mosaic=0.0,` | **关闭Mosaic！对小目标有害** |
| **98** | `cos_lr=False,` | `cos_lr=True,` | **启用余弦学习率调度** |
| **99** | `close_mosaic=10,` | `close_mosaic=0,` | mosaic已关闭，此参数无意义 |

---

### 🟡 推荐修改的参数（进一步优化）

| 行号 | 原始代码 | 修改为 | 说明 |
|------|----------|--------|------|
| **80** | `hsv_h=0.01,` | `hsv_h=0.01,` | 保持不变 |
| **81** | `hsv_s=0.5,` | `hsv_s=0.3,` | 降低饱和度增强 |
| **82** | `hsv_v=0.3,` | `hsv_v=0.3,` | 保持不变 |
| **83** | `degrees=10.0,` | `degrees=5.0,` | 减少旋转角度，保护小目标 |
| **84** | `translate=0.1,` | `translate=0.05,` | 减少平移 |
| **85** | `scale=0.3,` | `scale=0.2,` | 减少缩放范围 |
| **86** | `shear=5.0,` | `shear=0.0,` | 关闭剪切变换 |
| **87** | `perspective=0.0001,` | `perspective=0.0,` | 关闭透视变换 |

---

### 🟢 可选修改的参数

| 行号 | 原始代码 | 修改为 | 说明 |
|------|----------|--------|------|
| **69** | `lrf=0.02,` | `lrf=0.01,` | 最终学习率比例（可选） |
| **77** | `save_period=20,` | `save_period=50,` | 减少保存频率（可选） |

---

## 📝 完整修改后的代码片段

将 `D1_training.py` 第 52-119 行的 `model.train()` 部分替换为以下内容：

```python
    results = model.train(
        # === 数据集配置 ===
        data=str(config_file),    # 数据集配置文件
        
        # === 基础训练参数 ===
        epochs=epochs,            # 训练轮数
        batch=batch,              # 批大小
        imgsz=imgsz,              # 输入图像尺寸
        device=device,            # GPU设备
        
        # === 项目管理 ===
        project='runs/detect',    # 项目目录
        name=project_name,        # 实验名称
        exist_ok=True,            # 允许覆盖现有实验
        
        # === 学习率参数 (优化后) ===
        lr0=0.001,                # ⭐ 降低初始学习率
        lrf=0.01,                 # ⭐ 最终学习率比例
        momentum=0.937,           # 动量
        weight_decay=0.0005,      # 权重衰减
        warmup_epochs=5,          # ⭐ 增加warmup轮数
        warmup_momentum=0.8,      # 预热动量
        
        # === 早停和保存 ===
        patience=50,              # ⭐ 增加早停耐心值
        save_period=20,           # 每20轮保存一次
        
        # === 数据增强 (针对小目标优化) ===
        hsv_h=0.01,              # 色调增强 (很小)
        hsv_s=0.3,               # ⭐ 降低饱和度增强
        hsv_v=0.3,               # 明度增强
        degrees=5.0,             # ⭐ 减少旋转角度
        translate=0.05,          # ⭐ 减少平移
        scale=0.2,               # ⭐ 减少缩放
        shear=0.0,               # ⭐ 关闭剪切
        perspective=0.0,         # ⭐ 关闭透视
        fliplr=0.5,              # 水平翻转
        flipud=0.0,              # 垂直翻转 (关闭)
        mosaic=0.0,              # ⭐⭐⭐ 关闭Mosaic！
        mixup=0.0,               # MixUp (关闭)
        copy_paste=0.0,          # 复制粘贴 (关闭)
        
        # === 训练优化 ===
        amp=True,                # 自动混合精度训练
        cache=False,             # 不缓存图像
        rect=False,              # 不使用矩形训练
        cos_lr=True,             # ⭐⭐⭐ 启用余弦学习率调度！
        close_mosaic=0,          # mosaic已关闭
        
        # === 其他设置 ===
        workers=4,               # 数据加载线程数
        verbose=True,            # 详细输出
        seed=42,                 # 随机种子
        deterministic=True,      # 确定性训练
        single_cls=False,        # 多类别训练
        plots=True,              # 生成训练图表
        
        # === 验证设置 ===
        val=True,                # 训练时进行验证
        split='val',             # 验证集分割
        save_json=False,         # 不保存JSON结果
        save_hybrid=False,       # 不保存hybrid标签
    )
```

---

## 📊 Shell脚本参数修改

同时修改 `D1_training_all_models_v8_1280.sh` 中的参数：

| 行号 | 原始代码 | 修改为 |
|------|----------|--------|
| **23** | `PATIENCE=30` | `PATIENCE=50` |

---

## ⚠️ 重要说明

### 为什么关闭Mosaic？

Mosaic数据增强会将4张图像拼接成一张，这对于：
- ✅ 大目标检测很有效
- ❌ **小目标检测可能有害**，因为小目标可能被切割或模糊

对于D1孔洞检测（小目标密集），**强烈建议关闭Mosaic**。

### 为什么启用余弦学习率？

余弦学习率调度（`cos_lr=True`）会让学习率平滑下降：
- 训练初期：较高学习率快速学习
- 训练后期：学习率逐渐降低，精细调整

这比线性下降更稳定，能帮助模型更好地收敛。

### 关于patience参数

你提到训练集损失还在下降但验证集已经收敛导致早停。增加patience到50可以：
- 给模型更多机会找到更好的验证性能
- 避免过早停止训练

---

## 🔄 快速验证修改是否正确

修改完成后，可以用以下命令快速测试（只训练3个epoch）：

```bash
python D1_training.py \
    --model yolov8l.pt \
    --epochs 3 \
    --batch 16 \
    --imgsz 1280 \
    --device 0 \
    --patience 50 \
    --project-name D1_test_optimized
```

检查输出日志中是否显示：
- `cos_lr=True`
- `mosaic=0.0`
- `lr0=0.001`

---

## 📈 预期效果

| 改进项 | 预期提升 |
|--------|----------|
| 关闭Mosaic | +1-2% |
| 启用余弦学习率 | +0.5-1% |
| 降低学习率 | 更稳定收敛 |
| 增加patience | 避免早停 |
| **综合效果** | **+2-4%** |

从92%提升到94-96%是可行的。

