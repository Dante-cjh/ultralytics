# DOTA 数据切片功能使用指南

## 🎯 概述

ultralytics 框架内置了强大的 DOTA 数据集切片功能，无需从 mmrotate 移植！本指南详细介绍如何使用这些功能进行高效的数据预处理。

## 🚀 重要发现

**ultralytics 已经内置了完整的 DOTA 数据切片功能！**

- 📦 **内置模块**: `ultralytics.data.split_dota`
- ⚡ **功能完善**: 支持多尺度、自定义窗口、重叠度配置
- 🎨 **接口友好**: 比 mmrotate 更简洁易用
- 🔧 **高度可配置**: 支持多种切片策略

## 📁 文件结构

```
ultralytics/
├── dota_slice_tool.py           # 命令行工具
├── dota_slice_config.py         # 配置驱动工具
├── example_dota_slice.py        # 使用示例
├── configs/
│   └── slice_configs/           # 示例配置文件
│       ├── single_scale.json
│       ├── multi_scale.json
│       ├── high_overlap.json
│       └── custom_mmrotate_style.json
└── ultralytics/data/split_dota.py  # 核心切片模块（内置）
```

## 🛠️ 使用方法

### 方法 1: 直接使用内置功能

```python
from ultralytics.data.split_dota import split_trainval, split_test

# 基础切片
split_trainval(
    data_root="path/to/your/dota/data",
    save_dir="path/to/output",
    crop_size=1024,
    gap=200,
    rates=(1.0,)
)

# 多尺度切片
split_trainval(
    data_root="path/to/your/dota/data",
    save_dir="path/to/output",
    crop_size=1024,
    gap=200,
    rates=(0.5, 1.0, 1.5)  # 多尺度
)
```

### 方法 2: 使用命令行工具

```bash
# 基础切片
python dota_slice_tool.py \
    --data-root /path/to/dota/data \
    --save-dir /path/to/output \
    --crop-size 1024 \
    --gap 200

# 多尺度切片
python dota_slice_tool.py \
    --data-root /path/to/dota/data \
    --save-dir /path/to/output \
    --crop-size 1024 \
    --gap 200 \
    --rates 0.5 1.0 1.5
```

### 方法 3: 使用配置文件

```bash
# 创建示例配置文件
python dota_slice_config.py --create-samples

# 使用配置文件切片
python dota_slice_config.py --config configs/slice_configs/multi_scale.json
```

## ⚙️ 参数配置

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `crop_size` | 切片窗口大小 | 1024 | 小目标: 512, 大目标: 1024 |
| `gap` | 窗口重叠大小 | 200 | 小目标: 500, 大目标: 200 |
| `rates` | 多尺度缩放比例 | (1.0,) | 数据增强: (0.5, 1.0, 1.5) |
| `iof_threshold` | IoF 阈值 | 0.7 | 通常保持默认 |
| `img_rate_threshold` | 图像占比阈值 | 0.6 | 通常保持默认 |

## 📊 配置策略建议

### 🎯 小目标检测优化

```json
{
  "crop_size": 512,
  "gap": 500,
  "rates": [0.5, 1.0, 1.5],
  "description": "适合小目标，高重叠度"
}
```

### 🎯 大目标检测优化

```json
{
  "crop_size": 1024,
  "gap": 200,
  "rates": [1.0],
  "description": "适合大目标，标准设置"
}
```

### 🎯 数据增强策略

```json
{
  "crop_size": 1024,
  "gap": 200,
  "rates": [0.5, 1.0, 1.5],
  "description": "多尺度数据增强"
}
```

## 📂 数据目录结构

### 输入格式（DOTA 标准格式）

```
data_root/
├── images/
│   ├── train/
│   │   ├── image1.png
│   │   └── image2.png
│   └── val/
│       ├── image3.png
│       └── image4.png
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

### 输出格式（切片后）

```
save_dir/
├── images/
│   ├── train/
│   │   ├── image1__1024__0___0.jpg
│   │   ├── image1__1024__824___0.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1__1024__0___0.txt
    │   ├── image1__1024__824___0.txt
    │   └── ...
    └── val/
        └── ...
```

## 🎓 完整训练流程

### 1. 数据准备

```bash
# 假设您的数据在 dota_reorganized 目录
ls dota_reorganized/
# images/  labels/
```

### 2. 数据切片

```bash
python dota_slice_tool.py \
    --data-root dota_reorganized \
    --save-dir dota_sliced \
    --crop-size 1024 \
    --gap 200 \
    --rates 0.5 1.0 1.5
```

### 3. 训练模型

```bash
# 使用切片后的数据训练
yolo obb train \
    data=dota_sliced \
    model=yolo11n-obb.pt \
    epochs=100 \
    imgsz=1024 \
    batch=16
```

### 4. 训练脚本示例

```python
from ultralytics import YOLO
from ultralytics.data.split_dota import split_trainval

# 1. 数据切片
print("🔄 开始数据切片...")
split_trainval(
    data_root="dota_reorganized",
    save_dir="dota_sliced",
    crop_size=1024,
    gap=200,
    rates=(0.5, 1.0, 1.5)
)

# 2. 训练模型
print("🚀 开始模型训练...")
model = YOLO('yolo11n-obb.pt')
results = model.train(
    data='dota_sliced',
    epochs=100,
    imgsz=1024,
    batch=16,
    device=0
)
```

## 🔍 功能对比

| 功能 | mmrotate | ultralytics | 优势 |
|------|----------|-------------|------|
| 滑动窗口切片 | ✅ | ✅ | ultralytics 更简洁 |
| 多尺度支持 | ✅ | ✅ | 接口更友好 |
| IoF 计算 | ✅ | ✅ | 更高效的实现 |
| 多进程支持 | ✅ | ✅ | 内置进度条 |
| 配置文件支持 | ✅ | ✅ (通过工具) | JSON 格式更直观 |
| 背景图像处理 | ✅ | ✅ | 可配置 |

## 🎛️ 高级配置示例

### mmrotate 风格配置

```json
{
  "data_root": "/path/to/dota",
  "save_dir": "/path/to/output",
  "crop_sizes": [1024],
  "gaps": [500],
  "rates": [0.5, 1.0, 1.5],
  "include_test": false,
  "splits": ["train", "val"],
  "allow_background_images": true,
  "iof_threshold": 0.7,
  "img_rate_threshold": 0.6
}
```

### 自定义处理流程

```python
from ultralytics.data.split_dota import (
    load_yolo_dota, get_windows, get_window_obj, crop_and_save
)

# 自定义切片流程
def custom_slice_workflow(data_root, save_dir):
    # 加载数据
    annos = load_yolo_dota(data_root, split="train")
    
    for anno in annos:
        # 获取滑动窗口
        windows = get_windows(
            anno["ori_size"], 
            crop_sizes=(1024,), 
            gaps=(200,)
        )
        
        # 获取窗口内的目标
        window_objs = get_window_obj(anno, windows)
        
        # 裁剪并保存
        crop_and_save(
            anno, windows, window_objs, 
            im_dir=f"{save_dir}/images/train",
            lb_dir=f"{save_dir}/labels/train"
        )
```

## 🔧 故障排除

### 常见问题

1. **数据目录结构错误**
   ```bash
   # 检查目录结构
   python -c "
   from pathlib import Path
   data_root = Path('your_data_path')
   required = ['images/train', 'images/val', 'labels/train', 'labels/val']
   for p in required:
       print(f'{p}: {(data_root/p).exists()}')
   "
   ```

2. **内存不足**
   ```python
   # 减少并发进程数
   import os
   os.environ['NUM_THREADS'] = '4'  # 默认是 min(8, cpu_count())
   ```

3. **切片结果异常**
   ```python
   # 检查切片结果
   from ultralytics.data.split_dota import load_yolo_dota
   annos = load_yolo_dota('output_dir', split='train')
   print(f"切片后图像数量: {len(annos)}")
   ```

## 📝 总结

ultralytics 框架已经提供了非常完善的 DOTA 数据切片功能，无需从其他框架移植！主要优势：

1. **✅ 开箱即用**: 内置完整功能，无需额外安装
2. **🎯 功能完善**: 支持所有 mmrotate 的核心功能
3. **🚀 性能优异**: 更高效的实现和更好的用户体验
4. **🔧 易于集成**: 与 ultralytics 训练流程无缝集成
5. **📚 文档完整**: 官方文档和示例丰富

**建议**: 直接使用 ultralytics 内置功能，配合提供的工具脚本，可以高效完成 DOTA 数据的预处理和训练任务。

## 🎯 接下来的步骤

1. **准备数据**: 确保 DOTA 数据按标准格式组织
2. **选择策略**: 根据目标大小选择合适的切片参数
3. **执行切片**: 使用提供的工具进行数据切片
4. **开始训练**: 使用切片后的数据训练 YOLO-OBB 模型

---

*Happy slicing! 🎉*
