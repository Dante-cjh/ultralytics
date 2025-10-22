# Balloon 数据集切片训练指南

## 概述

本项目实现了将DOTA数据集的slice和multi-slice技术迁移到标准YOLO格式（水平框）数据集上的功能，并在Balloon数据集上进行了测试。

## 项目结构

```
ultralytics/
├── ultralytics/data/
│   ├── split_yolo.py              # YOLO格式数据切片工具（新增）
│   └── split_dota.py              # DOTA格式数据切片工具（原有）
├── balloon_training_with_slice.py        # 单尺度切片训练脚本
├── balloon_training_with_multi_slice.py  # 多尺度切片训练脚本
├── balloon_slice.yaml                    # 单尺度切片数据配置
└── balloon_multi_slice.yaml              # 多尺度切片数据配置
```

## 核心功能

### 1. split_yolo.py - YOLO格式数据切片工具

基于`split_dota.py`实现，但专门针对YOLO格式（水平框）数据：

**主要区别：**
- DOTA格式：使用8点旋转框 `[x1, y1, x2, y2, x3, y3, x4, y4]`
- YOLO格式：使用归一化中心点格式 `[class_id, x_center, y_center, width, height]`

**核心功能：**
- `load_yolo_format()`: 加载YOLO格式数据
- `bbox_iof_yolo()`: 计算水平框的IoF（Intersection over Foreground）
- `get_windows()`: 生成滑动窗口坐标
- `get_window_obj()`: 根据IoF阈值获取每个窗口中的目标
- `crop_and_save()`: 裁剪图像并保存新标签
- `split_trainval()`: 对训练集和验证集进行切片

### 2. 训练脚本

#### 单尺度切片训练 (`balloon_training_with_slice.py`)

```bash
# 完整流水线（切片 + 训练）
python balloon_training_with_slice.py \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_slice \
    --crop-size 640 \
    --gap 100 \
    --rates 1.0 \
    --epochs 100 \
    --batch 16 \
    --device 0

# 仅切片
python balloon_training_with_slice.py --slice-only \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_slice \
    --crop-size 640 \
    --gap 100

# 仅训练（需要先完成切片）
python balloon_training_with_slice.py --train-only \
    --epochs 100 \
    --batch 16 \
    --device 0
```

#### 多尺度切片训练 (`balloon_training_with_multi_slice.py`)

```bash
# 完整流水线（多尺度切片 + 训练）
python balloon_training_with_multi_slice.py \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_multi_slice \
    --crop-size 640 \
    --gap 100 \
    --rates 0.5 1.0 1.5 \
    --epochs 100 \
    --batch 16 \
    --device 0

# 仅多尺度切片
python balloon_training_with_multi_slice.py --slice-only \
    --data-root /home/cjh/mmdetection/data/balloon/yolo_format \
    --slice-dir /home/cjh/mmdetection/data/balloon/yolo_format_multi_slice \
    --crop-size 640 \
    --gap 100 \
    --rates 0.5 1.0 1.5
```

## 参数说明

### 切片参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--crop-size` | 切片窗口大小 | 640 |
| `--gap` | 窗口重叠大小 | 100 |
| `--rates` | 多尺度缩放比例 | 单尺度: [1.0]<br>多尺度: [0.5, 1.0, 1.5] |
| `--force-slice` | 强制重新切片 | False |

**缩放比例说明：**
- `rate = 0.5`: 窗口更大 (640/0.5 = 1280)，适合大目标
- `rate = 1.0`: 标准窗口 (640/1.0 = 640)
- `rate = 1.5`: 窗口更小 (640/1.5 ≈ 427)，适合小目标

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称或路径 | yolo11n.pt |
| `--epochs` | 训练轮数 | 100 |
| `--imgsz` | 输入图像尺寸 | 640 |
| `--batch` | 批次大小 | 16 |
| `--device` | GPU设备编号 | 0 |
| `--resume` | 恢复训练 | False |

## 数据集要求

数据集必须符合YOLO格式，目录结构如下：

```
data_root/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

标签格式（归一化坐标）：
```
class_id x_center y_center width height
0 0.671631 0.554730 0.401855 0.505201
0 0.742676 0.217682 0.345703 0.334324
```

## 切片效果

### 单尺度切片 (rate=1.0)

- 原始数据: 61张训练图像, 13张验证图像
- 切片后: 预计增加3-5倍图像数量
- 优点: 简单、快速
- 适用: 目标尺度相对统一的场景

### 多尺度切片 (rates=[0.5, 1.0, 1.5])

- 原始数据: 61张训练图像, 13张验证图像
- 切片后: 预计增加10-15倍图像数量
- 优点: 提高对不同尺度目标的检测能力
- 适用: 目标尺度变化较大的场景

## 技术细节

### IoF计算

使用IoF（Intersection over Foreground）而非IoU来判断目标是否应该包含在切片窗口中：

```python
IoF = Intersection(bbox, window) / Area(bbox)
```

- IoF阈值默认为0.7
- 当目标的70%以上在窗口内时，该目标会被包含

### 坐标转换

YOLO格式的坐标转换流程：

1. **读取标签**: `[class_id, x_center_norm, y_center_norm, w_norm, h_norm]`
2. **反归一化**: 乘以图像宽高得到绝对坐标
3. **转换为xyxy**: `[x_min, y_min, x_max, y_max]`
4. **计算IoF**: 判断目标是否在窗口内
5. **裁剪调整**: 减去窗口起始坐标
6. **重新归一化**: 除以窗口宽高
7. **保存标签**: 写入新的txt文件

## 与DOTA版本的对比

| 特性 | DOTA版本 | YOLO版本 |
|------|----------|----------|
| 数据格式 | OBB (8点旋转框) | HBB (水平框) |
| 标注格式 | `[cls, x1, y1, ..., x4, y4]` | `[cls, xc, yc, w, h]` |
| 任务类型 | OBB检测 | 目标检测 |
| 模型 | yolo11l-obb.pt | yolo11n.pt |
| IoF计算 | 使用shapely计算多边形 | 使用矩形交集 |
| 适用场景 | 遥感图像、倾斜目标 | 通用目标检测 |

## 测试运行

### 快速测试（仅切片）

```bash
# 单尺度切片测试
python balloon_training_with_slice.py --slice-only

# 多尺度切片测试
python balloon_training_with_multi_slice.py --slice-only
```

### 完整流程测试

```bash
# 单尺度：切片 + 训练
python balloon_training_with_slice.py \
    --epochs 50 \
    --batch 16 \
    --device 0

# 多尺度：切片 + 训练
python balloon_training_with_multi_slice.py \
    --epochs 50 \
    --batch 16 \
    --device 0
```

## 输出结果

训练完成后，结果保存在：

```
runs/detect/
├── balloon_yolo11n_slice/          # 单尺度训练结果
│   ├── weights/
│   │   ├── best.pt                  # 最佳模型
│   │   └── last.pt                  # 最后一轮模型
│   ├── args.yaml                    # 训练参数
│   ├── results.csv                  # 训练指标
│   └── *.png                        # 训练曲线图
└── balloon_yolo11n_multi_slice/    # 多尺度训练结果
    └── ...
```

## 常见问题

### 1. 内存不足

```bash
# 减小batch size
python balloon_training_with_slice.py --batch 8

# 减小图像尺寸
python balloon_training_with_slice.py --imgsz 512
```

### 2. 切片数据已存在

```bash
# 使用已有切片数据
python balloon_training_with_slice.py --train-only

# 或强制重新切片
python balloon_training_with_slice.py --force-slice
```

### 3. 调整切片参数

```bash
# 更密集的切片（增加数据量）
python balloon_training_with_slice.py \
    --crop-size 512 \
    --gap 50

# 更稀疏的切片（减少数据量）
python balloon_training_with_slice.py \
    --crop-size 800 \
    --gap 200
```

## 扩展应用

这套切片方案可以应用于任何YOLO格式的大尺寸小目标检测任务：

- **航拍图像**: 人群检测、车辆检测
- **卫星图像**: 建筑物检测、船只检测
- **医学图像**: 细胞检测、病灶检测
- **工业检测**: 缺陷检测、零件检测

只需要将`--data-root`和`--slice-dir`指向你的数据集路径即可。

## 参考

- 原始DOTA切片代码: `ultralytics/data/split_dota.py`
- DOTA训练脚本: `dota_training_with_slice.py`, `dota_training_with_multi_slice.py`
- Ultralytics YOLO文档: https://docs.ultralytics.com/

