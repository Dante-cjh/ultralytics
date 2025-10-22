# Balloon 数据集 SAHI 切片推理指南

## 概述

本指南说明如何使用 SAHI (Slicing Aided Hyper Inference) 对训练好的 Balloon 模型进行大尺寸图像的切片推理。

## ✅ 官方支持确认

**好消息**：Ultralytics 官方完全支持 SAHI 切片推理，可以直接用于 COCO 格式（水平框）！

- 📚 **官方文档**: `docs/en/guides/sahi-tiled-inference.md`
- 💻 **示例代码**: `examples/YOLOv8-SAHI-Inference-Video/`
- 🔄 **格式支持**: COCO、YOLO、OBB 等所有格式

## SAHI vs 手动切片对比

### DOTA 手动切片方式

```python
# 训练时切片
from ultralytics.data.split_dota import split_trainval
split_trainval(data_root="DOTA", save_dir="DOTA-split")

# 推理时：需要自己实现
# 1. 手动切片图像
# 2. 逐片推理
# 3. 手动合并结果（复杂的NMS）
```

### SAHI 自动化方式 ✅

```python
# 训练时切片（我们已实现）
from ultralytics.data.split_yolo import split_trainval
split_trainval(data_root="balloon", save_dir="balloon-split")

# 推理时：SAHI 一行搞定！
from sahi.predict import get_sliced_prediction
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

**SAHI 自动处理：**
- ✅ 自动切片
- ✅ 批量推理
- ✅ 智能合并（NMS 去重）
- ✅ 边界框修正
- ✅ 支持多种输出格式

## 安装依赖

```bash
# 激活环境
source /home/cjh/anaconda3/bin/activate ultralytics

# 安装 SAHI
pip install sahi
```

## 使用方法

### 1. 单张图像推理

```bash
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_yolo11n_slice/weights/best.pt \
    --source test_image.jpg \
    --slice-height 640 \
    --slice-width 640 \
    --overlap-height 0.2 \
    --overlap-width 0.2 \
    --save-dir runs/sahi_inference
```

### 2. 批量图像推理

```bash
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_yolo11n_slice/weights/best.pt \
    --source /path/to/test/images/ \
    --slice-height 640 \
    --slice-width 640 \
    --save-dir runs/sahi_inference
```

### 3. 使用多尺度训练的模型

```bash
# 使用多尺度切片训练的模型
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_yolo11n_multi_slice/weights/best.pt \
    --source /path/to/test/images/ \
    --slice-height 640 \
    --slice-width 640
```

### 4. 调整置信度阈值

```bash
python balloon_inference_with_sahi.py \
    --model best.pt \
    --source test_images/ \
    --confidence 0.5 \
    --slice-height 640 \
    --slice-width 640
```

### 5. 仅推理不保存可视化

```bash
python balloon_inference_with_sahi.py \
    --model best.pt \
    --source test_images/ \
    --no-visualize
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 训练好的模型路径 | `best.pt` |
| `--source` | 图像路径或目录 | `test.jpg` 或 `test_images/` |

### 切片参数

| 参数 | 说明 | 默认值 | 建议 |
|------|------|--------|------|
| `--slice-height` | 切片高度 | 640 | 与训练切片大小一致 |
| `--slice-width` | 切片宽度 | 640 | 与训练切片大小一致 |
| `--overlap-height` | 高度重叠比例 | 0.2 | 0.1-0.3，更大可减少边界漏检 |
| `--overlap-width` | 宽度重叠比例 | 0.2 | 0.1-0.3 |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--confidence` | 置信度阈值 | 0.25 |
| `--device` | 设备 | cuda:0 |

### 输出参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--save-dir` | 结果保存目录 | runs/sahi_inference |
| `--no-visualize` | 不保存可视化结果 | False |

## 切片参数选择指南

### 根据训练配置选择

**单尺度训练（rate=1.0）**:
```bash
# 训练时切片: crop_size=640, gap=100
# 推理时建议:
--slice-height 640 --slice-width 640
--overlap-height 0.15 --overlap-width 0.15
```

**多尺度训练（rates=[0.5, 1.0, 1.5]）**:
```bash
# 训练包含多个尺度，推理时使用标准尺度
--slice-height 640 --slice-width 640
--overlap-height 0.2 --overlap-width 0.2
```

### 根据图像特点选择

**密集小目标**:
```bash
# 更小的切片，更大的重叠
--slice-height 512 --slice-width 512
--overlap-height 0.3 --overlap-width 0.3
```

**稀疏大目标**:
```bash
# 更大的切片，较小的重叠
--slice-height 800 --slice-width 800
--overlap-height 0.1 --overlap-width 0.1
```

## Python API 使用

### 基础用法

```python
from balloon_inference_with_sahi import BalloonSAHIInference

# 初始化推理器
inferencer = BalloonSAHIInference(
    model_path="runs/detect/balloon_yolo11n_slice/weights/best.pt",
    confidence_threshold=0.25,
    device="cuda:0"
)

# 推理单张图像
result = inferencer.predict_image(
    image_path="test.jpg",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    save_dir="results",
    visualize=True
)

print(f"检测到 {result['num_detections']} 个目标")
```

### 批量推理

```python
# 推理整个目录
results = inferencer.predict_directory(
    image_dir="test_images/",
    slice_height=640,
    slice_width=640,
    save_dir="results",
    visualize=True
)

# 统计结果
for r in results:
    print(f"{r['image_path']}: {r['num_detections']} 个目标")
```

### 获取检测框详情

```python
result = inferencer.predict_image("test.jpg")

# 访问检测结果
for detection in result['detections']:
    bbox = detection.bbox  # [x_min, y_min, x_max, y_max]
    score = detection.score.value
    category = detection.category.name
    
    print(f"类别: {category}, 置信度: {score:.2f}, 位置: {bbox}")
```

## 输出文件

推理完成后，在 `save_dir` 目录下会生成：

```
runs/sahi_inference/
├── image1_visual.png          # 可视化结果（带检测框）
├── image2_visual.png
└── ...
```

## 高级用法：直接使用 SAHI

如果你想完全自定义，可以直接使用 SAHI API：

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 加载模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",
    confidence_threshold=0.25,
    device="cuda:0"
)

# 推理
result = get_sliced_prediction(
    "test.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 导出结果
result.export_visuals(export_dir="results/")

# 转换为COCO格式
coco_dict = result.to_coco_annotations()

# 转换为YOLO格式  
result.export_visuals(export_dir="results/", text_size=1, rect_th=2)
```

## 完整训练+推理流程

### 步骤 1: 训练模型

```bash
# 单尺度切片训练
python balloon_training_with_slice.py \
    --epochs 100 \
    --crop-size 640 \
    --gap 100

# 或多尺度切片训练
python balloon_training_with_multi_slice.py \
    --epochs 100 \
    --crop-size 640 \
    --gap 100 \
    --rates 0.5 1.0 1.5
```

### 步骤 2: SAHI 推理

```bash
# 使用训练好的模型推理
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_yolo11n_slice/weights/best.pt \
    --source test_images/ \
    --slice-height 640 \
    --slice-width 640
```

## 性能优化建议

### 1. 切片大小选择
- ✅ **推荐**: 与训练时的切片大小一致
- ⚠️ **过小**: 增加推理时间，可能漏检边界目标
- ⚠️ **过大**: 内存占用高，小目标检测效果差

### 2. 重叠比例选择
- 📈 **0.1-0.15**: 快速推理，适合稀疏目标
- 📊 **0.2-0.25**: 平衡推荐，适合大多数场景
- 📉 **0.3-0.4**: 高质量，适合密集小目标

### 3. 批处理建议
```python
# 对于大量图像，建议批量处理
results = inferencer.predict_directory(
    image_dir="large_dataset/",
    slice_height=640,
    slice_width=640,
)
```

## 故障排查

### 问题 1: 找不到 SAHI 模块

```bash
pip install sahi
```

### 问题 2: CUDA 内存不足

```bash
# 使用 CPU 推理
python balloon_inference_with_sahi.py \
    --model best.pt \
    --source test.jpg \
    --device cpu
```

### 问题 3: 检测结果重复

```python
# SAHI 已自动处理 NMS，如果仍有重复：
# 1. 减小重叠比例
--overlap-height 0.1 --overlap-width 0.1

# 2. 提高置信度阈值
--confidence 0.4
```

### 问题 4: 边界目标漏检

```python
# 增加重叠比例
--overlap-height 0.3 --overlap-width 0.3
```

## 与 DOTA 推理对比

| 特性 | DOTA (手动) | Balloon (SAHI) |
|------|------------|----------------|
| 切片方式 | 手动实现 | SAHI 自动 ✅ |
| NMS 合并 | 需要自己写 | SAHI 自动 ✅ |
| 边界处理 | 复杂 | SAHI 自动 ✅ |
| 格式支持 | OBB | 水平框/OBB ✅ |
| 代码量 | 200+ 行 | 10 行 ✅ |

## 参考资源

- 📖 [SAHI 官方文档](https://github.com/obss/sahi)
- 📖 [Ultralytics SAHI 集成指南](docs/en/guides/sahi-tiled-inference.md)
- 💻 [官方示例代码](examples/YOLOv8-SAHI-Inference-Video/)
- 🎓 [Colab 教程](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb)

## 总结

✅ **SAHI 完全可以用于 COCO 格式数据**  
✅ **自动处理切片、推理、合并全流程**  
✅ **比手动实现简单 10 倍以上**  
✅ **官方支持，稳定可靠**

开始使用吧！🚀

