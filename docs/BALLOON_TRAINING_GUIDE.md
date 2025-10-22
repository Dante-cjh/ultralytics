# Balloon 数据集训练指南

## 📋 目录

- [概述](#概述)
- [文件说明](#文件说明)
- [运行方式](#运行方式)
- [查看日志](#查看日志)
- [结果文件说明](#结果文件说明)
- [常见问题](#常见问题)

---

## 📖 概述

本项目提供了三种训练方式来训练 Balloon 数据集检测器：

1. **普通训练**：直接使用原始数据训练
2. **单尺度切片训练**：使用单一尺度切片后的数据训练
3. **多尺度切片训练**：使用多尺度切片后的数据训练

每种训练方式都会依次训练 `YOLO11m`、`YOLO11l`、`YOLO11x` 三个模型，并在验证集和测试集上进行 SAHI 推理。

---

## 📁 文件说明

### 训练脚本

| 文件名 | 说明 | 是否包含patience | TensorBoard日志 |
|--------|------|-----------------|----------------|
| `balloon_training.py` | 普通训练脚本 | ✅ (可配置) | ✅ |
| `balloon_training_with_slice.py` | 单尺度切片训练脚本 | ✅ (可配置) | ✅ |
| `balloon_training_with_multi_slice.py` | 多尺度切片训练脚本 | ✅ (可配置) | ✅ |

### 推理脚本

| 文件名 | 说明 | 适用场景 |
|--------|------|---------|
| `balloon_inference.py` | 普通YOLO推理脚本 | 未切片训练的模型 |
| `balloon_inference_with_sahi.py` | SAHI切片推理脚本 | 切片训练的模型或大尺寸图像 |

### Bash 自动化脚本

| 文件名 | 说明 | 训练模型 | 推理方式 |
|--------|------|---------|---------|
| `balloon_training_all_models.sh` | 普通训练自动化脚本 | yolo11m, yolo11l, yolo11x | 普通推理 |
| `balloon_training_slice_all_models.sh` | 单尺度切片训练自动化脚本 | yolo11m, yolo11l, yolo11x | SAHI推理 |
| `balloon_training_multi_slice_all_models.sh` | 多尺度切片训练自动化脚本 | yolo11m, yolo11l, yolo11x | SAHI推理 |

---

## 🚀 运行方式

### 方式一：普通训练（不切片）

直接使用原始数据训练三个模型（yolo11m, yolo11l, yolo11x）。

```bash
# 运行自动化脚本
bash balloon_training_all_models.sh

# 或者手动运行单个模型
python3 balloon_training.py --model yolo11m.pt --project-name balloon_yolo11m --epochs 200 --patience 20 --device 5
python3 balloon_training.py --model yolo11l.pt --project-name balloon_yolo11l --epochs 200 --patience 20 --device 5
python3 balloon_training.py --model yolo11x.pt --project-name balloon_yolo11x --epochs 200 --patience 20 --device 5
```

**训练参数：**
- 轮数 (epochs)：200
- 早停耐心值 (patience)：20
- 批次大小 (batch)：16
- 设备 (device)：GPU 5

**命名规则：**
- 项目名称：`balloon_{模型名称}_{时间戳}`
- 示例：`balloon_yolo11m_20241022_143000`

---

### 方式二：单尺度切片训练

使用单一尺度（1.0x）切片后的数据训练三个模型。

```bash
# 运行自动化脚本
bash balloon_training_slice_all_models.sh

# 或者手动运行
# 第一次：完整流程（包括切片）
python3 balloon_training_with_slice.py \
    --model yolo11m.pt \
    --project-name balloon_yolo11m_slice \
    --epochs 200 \
    --patience 20 \
    --device 5

# 后续：仅训练（使用已切片数据）
python3 balloon_training_with_slice.py \
    --model yolo11l.pt \
    --project-name balloon_yolo11l_slice \
    --epochs 200 \
    --patience 20 \
    --device 5 \
    --train-only

python3 balloon_training_with_slice.py \
    --model yolo11x.pt \
    --project-name balloon_yolo11x_slice \
    --epochs 200 \
    --patience 20 \
    --device 5 \
    --train-only
```

**切片参数：**
- 窗口大小 (crop_size)：640x640
- 重叠大小 (gap)：100
- 缩放比例 (rates)：1.0

**命名规则：**
- 项目名称：`balloon_{模型名称}_slice_{时间戳}`
- 示例：`balloon_yolo11m_slice_20241022_143000`

---

### 方式三：多尺度切片训练

使用多尺度（0.5x, 1.0x, 1.5x）切片后的数据训练三个模型。

```bash
# 运行自动化脚本
bash balloon_training_multi_slice_all_models.sh

# 或者手动运行
# 第一次：完整流程（包括多尺度切片）
python3 balloon_training_with_multi_slice.py \
    --model yolo11m.pt \
    --project-name balloon_yolo11m_multi_slice \
    --epochs 200 \
    --patience 20 \
    --device 5

# 后续：仅训练（使用已切片数据）
python3 balloon_training_with_multi_slice.py \
    --model yolo11l.pt \
    --project-name balloon_yolo11l_multi_slice \
    --epochs 200 \
    --patience 20 \
    --device 5 \
    --train-only

python3 balloon_training_with_multi_slice.py \
    --model yolo11x.pt \
    --project-name balloon_yolo11x_multi_slice \
    --epochs 200 \
    --patience 20 \
    --device 5 \
    --train-only
```

**切片参数：**
- 窗口大小 (crop_size)：640x640
- 重叠大小 (gap)：100
- 缩放比例 (rates)：0.5, 1.0, 1.5

**命名规则：**
- 项目名称：`balloon_{模型名称}_multi_slice_{时间戳}`
- 示例：`balloon_yolo11m_multi_slice_20241022_143000`

---

## 📊 查看日志

### TensorBoard 日志

所有训练脚本都会自动生成 TensorBoard 日志，保存在训练输出目录中。

#### 查看所有训练日志

```bash
# 查看所有训练运行的日志
tensorboard --logdir runs/detect/

# 在浏览器中访问
# http://localhost:6006
```

#### 查看特定模型日志

```bash
# 查看特定模型的日志（替换为实际的项目名称）
tensorboard --logdir runs/detect/balloon_yolo11m_20241022_143000/

# 指定端口
tensorboard --logdir runs/detect/balloon_yolo11m_20241022_143000/ --port 6007
```

#### 比较不同模型

```bash
# 同时查看多个模型的日志进行比较
tensorboard --logdir_spec \
  yolo11m:runs/detect/balloon_yolo11m_20241022_143000,\
  yolo11l:runs/detect/balloon_yolo11l_20241022_143000,\
  yolo11x:runs/detect/balloon_yolo11x_20241022_143000
```

### TensorBoard 显示内容

TensorBoard 会显示以下信息：

- **Scalars（标量）**：
  - 训练损失（box_loss, cls_loss, dfl_loss）
  - 验证指标（mAP@0.5, mAP@0.5:0.95, Precision, Recall）
  - 学习率变化
  
- **Images（图像）**：
  - 训练样本可视化
  - 验证预测结果
  - 混淆矩阵
  
- **Histograms（直方图）**：
  - 模型权重分布

### 训练日志文件位置

每个训练运行都会在项目目录下生成日志文件：

```
runs/detect/{项目名称}/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最后一轮模型权重
├── results.csv          # 训练结果CSV文件
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── F1_curve.png         # F1曲线
├── P_curve.png          # Precision曲线
├── R_curve.png          # Recall曲线
├── PR_curve.png         # PR曲线
└── events.out.tfevents.* # TensorBoard事件文件
```

---

## 📂 结果文件说明

### 目录结构

运行完成后，会生成以下目录结构：

```
ultralytics/
├── runs/
│   ├── detect/                          # 训练结果
│   │   ├── balloon_yolo11m_{timestamp}/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   ├── results.csv
│   │   │   ├── results.png
│   │   │   └── ...
│   │   ├── balloon_yolo11l_{timestamp}/
│   │   └── balloon_yolo11x_{timestamp}/
│   │
│   ├── inference/                       # 普通推理结果（未切片模型）
│   │   ├── balloon_yolo11m_{timestamp}_val/    # 验证集推理
│   │   │   ├── {image_name}.jpg        # 可视化图像
│   │   │   ├── labels/
│   │   │   │   └── {image_name}.txt    # 检测结果标签
│   │   │   └── ...
│   │   └── balloon_yolo11m_{timestamp}_test/   # 测试集推理
│   │
│   └── sahi_inference/                  # SAHI推理结果（切片模型）
│       ├── balloon_yolo11m_slice_{timestamp}_val/
│       │   ├── {image_name}_visual.jpg
│       │   └── ...
│       └── balloon_yolo11m_multi_slice_{timestamp}_val/
│
└── balloon_training.log                 # 训练日志（如果使用bash脚本）
```

### 文件说明

#### 训练结果文件

- **`weights/best.pt`**: 验证集上表现最好的模型权重
- **`weights/last.pt`**: 最后一轮训练的模型权重
- **`results.csv`**: 每个epoch的详细指标（CSV格式）
- **`results.png`**: 训练曲线图（损失、mAP等）
- **`confusion_matrix.png`**: 混淆矩阵
- **`*_curve.png`**: 各种评估曲线

#### 普通推理结果（未切片模型）

- **`{image_name}.jpg`**: 带检测框的可视化图像
- **`labels/{image_name}.txt`**: 检测结果标签文件（YOLO格式）
- 每行：`class_id confidence x_center y_center width height`

#### SAHI推理结果（切片模型）

- **`{image_name}_visual.jpg`**: 带检测框的可视化图像
- 每张图像一个文件，包含检测框、类别和置信度

---

## 🔍 监控训练进度

### 实时查看训练输出

```bash
# 如果使用bash脚本，可以实时查看输出
tail -f nohup.out

# 或者使用tee保存日志
bash balloon_training_all_models.sh 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt
```

### 查看GPU使用情况

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或使用gpustat
watch -n 1 gpustat
```

### 查看训练进度

训练过程中会输出以下信息：

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/200     1.23G      1.234      0.567      1.123         45        640: 100%|███
```

- **Epoch**: 当前轮数 / 总轮数
- **GPU_mem**: GPU内存使用
- **box_loss**: 边界框损失
- **cls_loss**: 分类损失
- **dfl_loss**: 分布焦点损失
- **Instances**: 当前批次的目标数量
- **Size**: 输入图像尺寸

---

## ⚙️ 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolo11n.pt | 模型名称或路径 |
| `--epochs` | 200 | 训练轮数 |
| `--batch` | 16 | 批次大小 |
| `--device` | 5 | GPU设备编号 |
| `--patience` | 20 | 早停耐心值 |
| `--project-name` | balloon_exp | 项目名称 |

### 切片参数（仅切片训练）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--crop-size` | 640 | 切片窗口大小 |
| `--gap` | 100 | 窗口重叠大小 |
| `--rates` | [1.0] 或 [0.5, 1.0, 1.5] | 多尺度缩放比例 |

### SAHI推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence` | 0.25 | 置信度阈值 |
| `--slice-height` | 640 | 切片高度 |
| `--slice-width` | 640 | 切片宽度 |
| `--overlap-height` | 0.2 | 高度重叠比例 |
| `--overlap-width` | 0.2 | 宽度重叠比例 |

---

## ❓ 常见问题

### 1. 如何修改GPU设备？

编辑bash脚本中的 `DEVICE` 变量：

```bash
# 修改为GPU 0
DEVICE=0
```

或者手动运行时指定：

```bash
python3 balloon_training.py --device 0 ...
```

### 2. 训练中断了怎么办？

使用 `--resume` 参数恢复训练：

```bash
python3 balloon_training.py --model yolo11m.pt --project-name balloon_yolo11m --resume
```

### 3. 如何调整批次大小？

如果GPU内存不足，可以减小批次大小：

```bash
# 修改bash脚本中的BATCH变量
BATCH=8

# 或者手动运行时指定
python3 balloon_training.py --batch 8 ...
```

### 4. 如何只运行特定模型？

注释掉bash脚本中不需要的模型：

```bash
# 只训练yolo11m
declare -a MODELS=("yolo11m.pt")
declare -a MODEL_NAMES=("yolo11m")
```

或者直接手动运行单个模型。

### 5. 切片数据已存在，如何重新切片？

使用 `--force-slice` 参数：

```bash
python3 balloon_training_with_slice.py --force-slice ...
```

### 6. 如何查看某个模型的验证结果？

```bash
# 查看results.csv
cat runs/detect/balloon_yolo11m_20241022_143000/results.csv

# 或者使用Python
python3 -c "import pandas as pd; print(pd.read_csv('runs/detect/balloon_yolo11m_20241022_143000/results.csv'))"
```

### 7. TensorBoard无法访问？

确保端口没有被占用，或者指定其他端口：

```bash
tensorboard --logdir runs/detect/ --port 6007
```

### 8. 如何比较不同训练方式的效果？

使用TensorBoard的多实验比较功能：

```bash
tensorboard --logdir_spec \
  普通训练:runs/detect/balloon_yolo11m_20241022_143000,\
  单尺度切片:runs/detect/balloon_yolo11m_slice_20241022_143000,\
  多尺度切片:runs/detect/balloon_yolo11m_multi_slice_20241022_143000
```

---

## 📈 性能优化建议

### 训练速度优化

1. **增大批次大小**（如果GPU内存允许）：
   ```bash
   --batch 32
   ```

2. **使用AMP（自动混合精度）**：
   已默认启用（`amp=True`）

3. **减少数据增强**（如果训练过慢）：
   修改训练脚本中的数据增强参数

### 训练效果优化

1. **调整学习率**：
   修改训练脚本中的 `lr0` 和 `lrf` 参数

2. **增加训练轮数**：
   ```bash
   --epochs 300
   ```

3. **调整早停耐心值**：
   ```bash
   --patience 30
   ```

---

## 📝 检查清单

### 运行前检查

- [ ] 确认数据集路径正确
- [ ] 确认GPU可用：`nvidia-smi`
- [ ] 确认Python环境正确
- [ ] 确认所需包已安装：`ultralytics`, `sahi`
- [ ] 确认磁盘空间充足（至少50GB）

### 运行中监控

- [ ] 监控GPU使用情况
- [ ] 监控训练损失是否下降
- [ ] 监控验证mAP是否提升
- [ ] 检查TensorBoard日志

### 运行后检查

- [ ] 确认模型文件已生成（best.pt）
- [ ] 确认推理结果已生成
- [ ] 查看训练曲线图（results.png）
- [ ] 对比不同模型的性能

---

## 📧 联系方式

如有问题，请参考：
- Ultralytics官方文档：https://docs.ultralytics.com
- SAHI文档：https://github.com/obss/sahi

---

**祝训练顺利！🎉**

