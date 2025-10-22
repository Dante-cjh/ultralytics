# Balloon 数据集 - 快速开始指南

## 🎯 一键运行完整流程

```bash
# 直接运行 bash 脚本，完成：切片 → 训练 → 推理
./balloon_complete_pipeline.sh
```

**流程说明：**
1. ✂️ **数据切片**：将 Balloon 数据集切分为 640x640 的小片
2. 🚀 **模型训练**：训练 50 个 epoch（约 5-10 分钟）
3. 🔍 **SAHI 推理**：对验证集进行切片推理并可视化

**预期结果：**
- 切片数据：`/home/cjh/mmdetection/data/balloon/yolo_format_slice/`
- 训练模型：`runs/detect/balloon_demo/weights/best.pt`
- 推理结果：`runs/balloon_demo_inference/` (带检测框的图像)

---

## 🔧 单独运行各个步骤

### 1. 仅数据切片

```bash
python balloon_training_with_slice.py --slice-only \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_slice \
    --crop-size 640 --gap 100
```

### 2. 仅模型训练

```bash
python balloon_training_with_slice.py --train-only \
    --model yolo11n.pt \
    --epochs 50 \
    --batch 16 \
    --device 0
```

### 3. 仅 SAHI 推理

```bash
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_demo/weights/best.pt \
    --source /home/cjh/mmdetection/data/balloon/yolo_format/images/val/ \
    --slice-height 640 \
    --slice-width 640 \
    --save-dir runs/inference_results
```

---

## 🐛 问题修复记录

### ✅ 已修复：推理结果没有检测框

**问题**：SAHI 0.11.14 的 `export_visuals()` 方法有 bug，不会绘制检测框

**解决方案**：修改为手动绘制检测框
- 文件：`balloon_inference_with_sahi.py`
- 修复位置：第 117-152 行
- 现在会正确绘制绿色检测框和标签

**验证方法**：
```python
from PIL import Image
import numpy as np

orig = Image.open("原图.jpg")
result = Image.open("推理结果.jpg")
diff = np.abs(np.array(orig) - np.array(result)).sum()
print(f"像素差异: {diff:,.0f}")  # 应该 > 1,000,000
```

---

## 📊 预期效果

### 数据切片效果
- **原始数据**：61 训练图像，13 验证图像
- **切片后**：446 训练图像（7.3x），92 验证图像（7x）
- **切片参数**：640x640，重叠 100 像素

### 训练效果（50 epochs）
- **模型**：YOLO11n
- **训练时间**：约 5-10 分钟（RTX 4090）
- **预期 mAP50**：0.5-0.7（根据数据质量）

### 推理效果
- **切片数量**：
  - 大图 (2048x2048)：12 个切片
  - 小图 (1024x1024)：4 个切片
- **检测结果**：带绿色边界框和置信度标签的图像
- **平均检测数**：约 4-5 个气球/图像

---

## 🔍 查看结果

### 1. 查看推理结果图像

```bash
ls -lh runs/balloon_demo_inference/
```

### 2. 查看训练指标

```bash
cat runs/detect/balloon_demo/results.csv
```

### 3. 启动 TensorBoard

```bash
tensorboard --logdir runs/detect/balloon_demo
```

---

## ⚙️ 参数调整

### 切片参数
- `--crop-size 640`：切片大小（建议与训练一致）
- `--gap 100`：重叠大小（100-200 推荐）
- `--overlap-height 0.2`：推理时的重叠比例（0.15-0.3）

### 训练参数
- `--epochs 50`：训练轮数（50-100 推荐）
- `--batch 16`：批次大小（根据 GPU 调整）
- `--imgsz 640`：输入图像尺寸

### 推理参数
- `--confidence 0.25`：置信度阈值（0.2-0.5）
- `--device 0`：GPU 设备（或 `cpu`）

---

## 📝 文件说明

| 文件 | 说明 |
|------|------|
| `balloon_training_with_slice.py` | 单尺度切片训练脚本 |
| `balloon_training_with_multi_slice.py` | 多尺度切片训练脚本 |
| `balloon_inference_with_sahi.py` | SAHI 切片推理脚本 |
| `balloon_complete_pipeline.sh` | 一键完整流程脚本 |
| `balloon_slice.yaml` | 单尺度数据配置 |
| `balloon_multi_slice.yaml` | 多尺度数据配置 |

---

## 🚀 扩展应用

这套方案可以应用到任何 YOLO 格式的大尺寸小目标检测任务：

1. 修改数据路径
2. 调整切片参数
3. 运行训练+推理

**示例场景**：
- 航拍图像（人群、车辆检测）
- 卫星图像（建筑物、船只检测）
- 医学图像（细胞、病灶检测）
- 工业检测（缺陷、零件检测）

---

## 💡 最佳实践

1. **首次运行**：使用默认参数先跑通流程
2. **调优切片**：根据目标大小调整 `crop_size` 和 `gap`
3. **调优训练**：增加 `epochs` 到 100-200
4. **调优推理**：调整 `overlap` 和 `confidence`

---

## 📞 问题排查

### 训练很慢
```bash
# 减小批次大小
--batch 8

# 减小图像尺寸
--imgsz 512
```

### 内存不足
```bash
# 使用 CPU
--device cpu

# 或减小批次
--batch 4
```

### 检测效果差
```bash
# 增加训练轮数
--epochs 100

# 调整置信度阈值
--confidence 0.3

# 增加推理重叠
--overlap-height 0.3 --overlap-width 0.3
```

---

**开始使用吧！** 🎈

