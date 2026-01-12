# 多尺度推理指南

## 📖 概述

本指南介绍了三种不同的高级推理方法，帮助你提升模型在测试集上的检测性能。

## 🔍 三种推理方法对比

### 1. **标准推理** (`balloon_inference.py`)
- **原理**: 使用单一尺度进行推理
- **速度**: ⭐⭐⭐⭐⭐ 最快
- **精度**: ⭐⭐⭐ 基准
- **适用场景**: 快速推理、实时应用

```bash
python3 balloon_inference.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --imgsz 1280 \
    --confidence 0.25 \
    --iou 0.5
```

---

### 2. **多尺度推理** (`balloon_inference_multiscale.py`) ⭐ 新增
- **原理**: 在多个尺度上推理，然后融合结果
  - 对同一张图像使用不同的输入尺寸(如640, 832, 1024, 1280)
  - 每个尺度独立推理，得到各自的检测结果
  - 使用NMS或WBF融合所有尺度的检测框
  
- **优势**:
  - 🎯 提高检测召回率 - 不同尺度捕获不同大小的目标
  - 📊 提升定位精度 - 多尺度信息融合
  - 🔧 灵活配置 - 可自定义尺度和融合方法
  
- **速度**: ⭐⭐⭐ 中等 (约为标准推理的3-4倍)
- **精度**: ⭐⭐⭐⭐⭐ 最高 (通常比单尺度高2-5% mAP)
- **适用场景**: 
  - 测试集评估（追求最高精度）
  - 目标尺度变化大的场景
  - 对小目标检测要求高

#### 使用示例：

```bash
# NMS融合 (默认)
python3 balloon_inference_multiscale.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --scales 640 832 1024 1280 \
    --fusion nms \
    --confidence 0.25 \
    --iou 0.5 \
    --device cuda:0 \
    --save-dir runs/multiscale_inference

# WBF融合 (更好的框融合)
python3 balloon_inference_multiscale.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --scales 640 832 1024 1280 \
    --fusion wbf \
    --confidence 0.25 \
    --iou 0.45 \
    --device cuda:0

# 快速多尺度 (只用2-3个尺度)
python3 balloon_inference_multiscale.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --scales 832 1280 \
    --fusion nms
```

#### 参数说明：
- `--scales`: 推理尺度列表，可以是任意数量的尺度
- `--fusion`: 融合方法
  - `nms`: Non-Maximum Suppression (默认，速度快)
  - `wbf`: Weighted Boxes Fusion (精度更高，需要安装 `pip install ensemble-boxes`)
- `--iou`: IoU阈值，控制重叠框的融合程度
  - NMS: 推荐 0.5-0.6
  - WBF: 推荐 0.4-0.5 (WBF对IoU阈值更敏感)

---

### 3. **SAHI切片推理** (`balloon_inference_with_sahi.py`)
- **原理**: 将大图切成小块，分别推理后融合
  - 将图像切分成多个重叠的小块(如640x640)
  - 对每个小块独立推理
  - 使用NMS融合所有小块的检测结果
  
- **优势**:
  - 🖼️ 处理超大分辨率图像 - 不受GPU显存限制
  - 🔍 检测小目标 - 切片后小目标占比更大
  - 📈 提高密集场景检测 - 减少遮挡影响
  
- **速度**: ⭐⭐ 较慢 (取决于切片数量)
- **精度**: ⭐⭐⭐⭐ 高 (特别是小目标和大图)
- **适用场景**: 
  - 超高分辨率图像 (如4K, 8K)
  - 小目标密集的场景 (如遥感图像、监控视频)
  - GPU显存有限时

```bash
python3 balloon_inference_with_sahi.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --slice-height 640 \
    --slice-width 640 \
    --overlap-height 0.15 \
    --overlap-width 0.15 \
    --confidence 0.25
```

---

## 📊 性能对比

| 方法 | 推理速度 | mAP提升 | 小目标AP | 适用场景 | GPU显存 |
|------|---------|---------|----------|----------|---------|
| 标准推理 | 1.0x (基准) | 0% | 基准 | 通用 | 中等 |
| 多尺度推理 | 0.25-0.3x | +2~5% | ++++ | 测试评估 | 较高 |
| SAHI切片 | 0.2-0.5x | +3~8% | +++++ | 大图/小目标 | 较低 |

*注: 速度相对于标准推理，mAP提升取决于数据集特性*

---

## 🎯 融合方法详解

### NMS (Non-Maximum Suppression)
- **原理**: 抑制重叠的低置信度框
- **流程**:
  1. 按置信度排序
  2. 保留置信度最高的框
  3. 删除与其IoU > 阈值的其他框
  4. 重复直到处理完所有框
  
- **优点**: 快速、简单、稳定
- **缺点**: 可能丢失有效检测（如重叠目标）

### WBF (Weighted Boxes Fusion)
- **原理**: 融合重叠框而非删除
- **流程**:
  1. 将所有框按位置聚类
  2. 对每个聚类，使用加权平均融合坐标
  3. 置信度为所有框置信度的加权平均
  
- **优点**: 
  - 更好的定位精度
  - 不丢失有效信息
  - 置信度估计更准确
  
- **缺点**: 
  - 稍慢（仍然很快）
  - 需要额外依赖包

**推荐**: 
- 如果追求最高精度 → 使用 WBF
- 如果需要平衡速度和精度 → 使用 NMS

---

## 🔧 高级配置

### 1. 自定义尺度选择

不同尺度适合不同目标大小：
- **640**: 适合中等大小目标
- **832**: 平衡小目标和大目标
- **1024**: 适合小目标
- **1280**: 适合超小目标

**建议组合**:
```bash
# 快速 (2个尺度)
--scales 832 1280

# 平衡 (3个尺度)  
--scales 640 1024 1280

# 精度优先 (4个尺度)
--scales 640 832 1024 1280

# 极限精度 (5个尺度)
--scales 512 640 832 1024 1280
```

### 2. 置信度和IoU阈值调优

```bash
# 高召回率 (Recall)
--confidence 0.15 --iou 0.4

# 平衡
--confidence 0.25 --iou 0.5

# 高精确率 (Precision)
--confidence 0.35 --iou 0.6
```

### 3. 结合TTA (Test-Time Augmentation)

Ultralytics内置的TTA可以与多尺度推理结合：

```python
# 在代码中启用TTA
results = self.model.predict(
    source=image,
    imgsz=scale,
    augment=True,  # 启用TTA
    conf=self.confidence_threshold,
    ...
)
```

---

## 💡 实战建议

### 训练阶段
1. 使用标准推理快速验证
2. 使用 `--imgsz` 匹配训练尺寸

### 测试评估阶段
1. 使用多尺度推理获得最佳性能
2. 尝试不同的尺度组合
3. 对比NMS和WBF的效果

### 生产部署
1. 根据速度要求选择方法
2. 如果速度允许，使用2-3个尺度的多尺度推理
3. 对于超大图，使用SAHI

---

## 📝 完整工作流示例

```bash
# 步骤1: 标准推理（快速验证）
python3 balloon_inference.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --imgsz 1280

# 步骤2: 多尺度推理（提升性能）
python3 balloon_inference_multiscale.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --scales 832 1024 1280 \
    --fusion wbf \
    --save-dir runs/multiscale_wbf

# 步骤3: SAHI推理（如果图像很大或小目标多）
python3 balloon_inference_with_sahi.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source data/test/ \
    --slice-height 640 \
    --slice-width 640 \
    --save-dir runs/sahi

# 步骤4: 对比结果
# 分析三种方法的检测结果，选择最适合你数据集的方法
```

---

## 🛠️ 依赖安装

```bash
# 基础依赖（已包含在ultralytics中）
pip install ultralytics

# WBF支持（可选，用于更好的框融合）
pip install ensemble-boxes

# SAHI支持（可选，用于切片推理）
pip install sahi
```

---

## ❓ 常见问题

### Q1: 多尺度推理会慢多少？
A: 大约是单尺度的3-4倍。如果使用4个尺度，理论上是4倍，但有GPU并行优化。

### Q2: 什么时候应该用多尺度而不是SAHI？
A: 
- 多尺度: 图像大小适中(如1-2K)，想提升整体性能
- SAHI: 图像超大(如4K+)，或者小目标特别多

### Q3: NMS和WBF哪个更好？
A: WBF通常精度更高(+0.5-1% mAP)，但需要安装额外依赖。建议优先尝试WBF。

### Q4: 可以同时使用多尺度和SAHI吗？
A: 可以，但通常不必要，会非常慢。建议二选一。

### Q5: 训练时也要用多尺度吗？
A: 训练时已经有多尺度训练(`--mosaic`)。这里的多尺度推理是测试时的技巧。

---

## 📚 参考资料

- [Test-Time Augmentation (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/)
- [Model Ensembling](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling/)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [WBF Paper](https://arxiv.org/abs/1910.13302)

---

## 🎓 总结

| 如果你想... | 使用... |
|------------|---------|
| 快速测试 | 标准推理 |
| 获得最高精度 | 多尺度推理 + WBF |
| 处理超大图像 | SAHI切片推理 |
| 检测超小目标 | 多尺度推理 (增加大尺寸) |
| 平衡速度和精度 | 多尺度推理 (2-3个尺度) + NMS |

**推荐配置** (D1数据集):
```bash
python3 balloon_inference_multiscale.py \
    --model runs/detect/D1_yolo11m_1280/weights/best.pt \
    --source /path/to/test/images \
    --scales 832 1024 1280 \
    --fusion wbf \
    --confidence 0.25 \
    --iou 0.45 \
    --device cuda:0
```

这个配置在大多数情况下能获得最佳的精度提升，同时保持合理的推理速度！

