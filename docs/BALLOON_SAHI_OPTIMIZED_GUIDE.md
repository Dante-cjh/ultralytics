# Balloon 数据集 SAHI 优化推理指南

## 🎯 概述

本指南说明如何使用优化后的 SAHI 推理系统，解决检测框过多和重复检测的问题。

## 🚀 快速开始

### 1. 单尺度切片训练 + SAHI推理

```bash
# 运行单尺度切片训练（自动包含SAHI推理）
bash balloon_training_slice_all_models.sh
```

### 2. 多尺度切片训练 + SAHI推理

```bash
# 运行多尺度切片训练（自动包含SAHI推理）
bash balloon_training_multi_slice_all_models.sh
```

### 3. 单独SAHI推理

```bash
# 使用优化参数进行SAHI推理
python balloon_inference_with_sahi.py \
    --model runs/detect/balloon_yolo11n_slice/weights/best.pt \
    --source /path/to/test/images/ \
    --confidence 0.3 \
    --postprocess-type NMS \
    --postprocess-threshold 0.6 \
    --min-box-area 200 \
    --max-detections 50
```

## ⚙️ 关键优化参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--confidence` | 0.3 | 置信度阈值 | 0.3-0.5，越高过滤越严格 |
| `--postprocess-threshold` | 0.6 | NMS IoU阈值 | 0.5-0.8，越高去重越强 |
| `--min-box-area` | 200 | 最小检测框面积 | 100-500，过滤小框 |
| `--max-detections` | 50 | 最大检测数量 | 30-100，限制总数量 |
| `--overlap-height/width` | 0.15 | 重叠比例 | 0.1-0.2，与训练匹配 |

## 🔧 参数调优策略

### 问题：检测框过多
```bash
# 解决方案：提高置信度 + 限制数量
--confidence 0.4 --max-detections 30
```

### 问题：重复检测框
```bash
# 解决方案：提高NMS阈值
--postprocess-threshold 0.7
```

### 问题：小框干扰
```bash
# 解决方案：增加最小面积
--min-box-area 500
```

### 问题：边界漏检
```bash
# 解决方案：增加重叠率
--overlap-height 0.2 --overlap-width 0.2
```

## 📊 训练-推理对应关系

| 训练方式 | 推理方式 | 脚本文件 |
|---------|---------|----------|
| 单尺度切片 | SAHI推理 | `balloon_training_slice_all_models.sh` |
| 多尺度切片 | SAHI推理 | `balloon_training_multi_slice_all_models.sh` |

## 🎯 最佳实践

### 1. 参数匹配原则
- **重叠率**：推理时与训练时保持一致
- **切片大小**：推理时与训练时保持一致
- **置信度**：根据验证集表现调整

### 2. 性能优化
- 先用默认参数测试
- 根据结果调整置信度和NMS阈值
- 必要时调整检测框过滤参数

### 3. 结果验证
- 检查检测框数量是否合理
- 验证是否有重复检测
- 确认小目标检测效果

## 📁 输出结构

```
runs/
├── detect/                          # 训练结果
│   └── balloon_yolo11n_slice_*/    # 模型权重
└── sahi_inference/                  # SAHI推理结果
    └── balloon_yolo11n_slice_*_val/ # 验证集推理结果
        ├── image1_visual.jpg        # 可视化结果
        └── ...
```

## 🔍 故障排除

### 检测框仍然过多
1. 提高 `--confidence` 到 0.4-0.5
2. 降低 `--max-detections` 到 30
3. 增加 `--min-box-area` 到 500

### 重复检测框
1. 提高 `--postprocess-threshold` 到 0.7
2. 尝试 `--postprocess-type NMM`

### 漏检小目标
1. 降低 `--confidence` 到 0.25
2. 增加重叠率到 0.2
3. 减少 `--min-box-area` 到 100

## 📈 性能对比

| 方法 | 检测框数量 | 重复检测 | 小目标检测 | 推荐场景 |
|------|------------|----------|------------|----------|
| 原始SAHI | 过多 | 严重 | 好 | ❌ 不推荐 |
| 优化SAHI | 适中 | 轻微 | 好 | ✅ 推荐 |
| 普通推理 | 适中 | 无 | 一般 | 非切片模型 |

## 🎉 总结

通过优化SAHI推理参数，我们成功解决了：
- ✅ 检测框过多的问题
- ✅ 重复检测框的问题  
- ✅ 小目标检测效果保持
- ✅ 参数可调节性强

现在您可以使用优化后的系统获得更清洁、更准确的检测结果！
