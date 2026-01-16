# 两阶段级联检测系统 - 更新说明

## 概述

本次更新解决了以下4个问题，优化了两阶段级联检测系统的使用体验。

---

## 1. ✅ 批量推理和评估功能

### 问题
- 之前只有单张图像推理（`run_cascaded_detection.sh infer`）
- 缺少批量评估脚本

### 解决方案
创建了完整的批量评估系统：

#### 新增文件
- **`run_cascaded_eval.sh`** - 批量评估Shell脚本
- 增强 **`balloon_cascaded_infer_all.py`** - 添加可视化和标签保存功能

#### 功能特性
1. **批量推理**: 在整个验证集上运行两阶段检测
2. **性能对比**: 自动对比单阶段YOLO和两阶段级联的性能
3. **可视化保存**: 
   - 每张图像保存左右对比图（单阶段 vs 两阶段）
   - 保存路径: `runs/cascaded_eval_balloon/visualizations/`
4. **标签保存**: 
   - YOLO格式的标签文件
   - 单阶段: `runs/cascaded_eval_balloon/labels_single_stage/`
   - 两阶段: `runs/cascaded_eval_balloon/labels_two_stage/`
5. **详细报告**:
   - JSON格式的详细结果: `detailed_results.json`
   - 文本格式的评估报告: `evaluation_report.txt`

#### 使用方法

```bash
# 运行批量评估
bash run_cascaded_eval.sh

# 查看评估报告
cat runs/cascaded_eval_balloon/evaluation_report.txt

# 查看可视化结果
ls runs/cascaded_eval_balloon/visualizations/

# 查看标签文件
ls runs/cascaded_eval_balloon/labels_two_stage/
```

#### 输出结构
```
runs/cascaded_eval_balloon/
├── visualizations/          # 可视化图像（左右对比）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels_single_stage/     # 单阶段YOLO标签
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── labels_two_stage/        # 两阶段级联标签
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── detailed_results.json    # 详细结果（JSON格式）
└── evaluation_report.txt    # 评估报告（文本格式）
```

---

## 2. ✅ 数据集路径优化

### 问题
- 数据集保存在项目根目录，不便管理
- 不同YOLO模型生成的数据集会相互覆盖

### 解决方案

#### 自动命名规则
数据集现在自动保存到：
```
data/<model_name>_cascaded_data_balloon/
```

例如：
- YOLO模型: `runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt`
- 数据集路径: `data/balloon_yolo11l_20251203_160322_cascaded_data_balloon/`

#### 优点
1. **统一管理**: 所有数据集在 `data/` 目录下
2. **避免冲突**: 不同模型的数据集独立存储
3. **易于识别**: 路径包含模型名称和时间戳
4. **便于迁移**: 整个 `data/` 目录可以方便地备份/迁移

#### Shell脚本自动提取模型名称
```bash
# 从YOLO模型路径自动提取名称
YOLO_MODEL="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"
YOLO_MODEL_NAME=$(basename $(dirname $(dirname "$YOLO_MODEL")))
# 结果: balloon_yolo11l_20251203_160322

# 自动生成数据集路径
DATA_DIR="data/${YOLO_MODEL_NAME}_cascaded_data_balloon"
```

---

## 3. ✅ 数据准备跳过机制

### 问题
- 每次运行都会重新生成数据集
- 数据准备耗时较长（约20秒）
- 如果数据已存在，重复生成浪费时间

### 解决方案

#### 智能检测
脚本现在会自动检测数据集是否已存在：
1. 检查 `data_list.json` 和 `stats.json` 是否存在
2. 如果存在，显示统计信息并跳过
3. 如果不存在或使用 `--force`，则重新生成

#### 使用方法

```bash
# 默认：如果数据存在则跳过
bash run_cascaded_detection.sh prepare

# 强制重新生成
FORCE_PREPARE=true bash run_cascaded_detection.sh prepare

# 或者直接使用Python命令
python balloon_cascaded_detection.py prepare \
    --yolo-model ... \
    --force  # 强制重新生成
```

#### 输出示例

**数据已存在（跳过）:**
```
⏭️  train集数据已存在，跳过准备步骤
   数据路径: data/balloon_yolo11l_20251203_160322_cascaded_data_balloon/train
   如需重新生成，请使用 --force 参数
   总图像数: 61
   总候选框数: 4883
   正样本数: 263
   负样本数: 4620
```

**强制重新生成:**
```
🔍 处理 train 集...
准备train数据: 100%|██████████| 61/61 [00:20<00:00,  2.97it/s]

✅ train集准备完成!
   总图像数: 61
   总候选框数: 4883
   正样本数: 263
   负样本数: 4620
```

---

## 4. ✅ 训练权重路径优化

### 问题
- 训练权重保存路径不够清晰
- 无法区分不同YOLO模型训练的分类器

### 解决方案

#### 新的保存路径规则
```
runs/mobilenet/<yolo_model_name>_<timestamp>/
```

例如：
- YOLO模型: `balloon_yolo11l_20251203_160322`
- 训练时间: `20251209_124500`
- 保存路径: `runs/mobilenet/balloon_yolo11l_20251203_160322_20251209_124500/`

#### 路径结构
```
runs/mobilenet/
├── balloon_yolo11l_20251203_160322_20251209_124500/
│   ├── best.pt              # 最佳模型
│   ├── epoch_10.pt          # 第10轮
│   ├── epoch_20.pt          # 第20轮
│   └── ...
├── balloon_yolov8l_1280_20251209_150000/
│   └── ...
└── ...
```

#### 优点
1. **清晰追溯**: 从路径可以看出使用的YOLO模型
2. **时间标记**: 包含训练时间戳
3. **统一管理**: 所有分类器权重在 `runs/mobilenet/` 下
4. **避免覆盖**: 每次训练都有独立的时间戳目录

#### Shell脚本实现
```bash
# 自动生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 自动生成训练目录
TRAIN_DIR="runs/mobilenet/${YOLO_MODEL_NAME}_${TIMESTAMP}"
```

---

## 完整工作流程

### 1. 准备数据（只需运行一次）

```bash
# 第一次运行：生成数据
bash run_cascaded_detection.sh prepare

# 再次运行：自动跳过（数据已存在）
bash run_cascaded_detection.sh prepare

# 强制重新生成
FORCE_PREPARE=true bash run_cascaded_detection.sh prepare
```

### 2. 训练分类器

```bash
bash run_cascaded_detection.sh train

# 训练结果保存至:
# runs/mobilenet/balloon_yolo11l_20251203_160322_20251209_124500/best.pt
```

### 3. 单张图像推理（演示）

```bash
bash run_cascaded_detection.sh infer

# 结果保存至:
# runs/cascaded_infer_balloon/balloon_yolo11l_20251203_160322/
```

### 4. 批量评估（推荐）

```bash
bash run_cascaded_eval.sh

# 结果包括:
# - 可视化图像: runs/cascaded_eval_balloon/visualizations/
# - 单阶段标签: runs/cascaded_eval_balloon/labels_single_stage/
# - 两阶段标签: runs/cascaded_eval_balloon/labels_two_stage/
# - 详细结果: runs/cascaded_eval_balloon/detailed_results.json
# - 评估报告: runs/cascaded_eval_balloon/evaluation_report.txt
```

### 5. 查看结果

```bash
# 查看评估报告
cat runs/cascaded_eval_balloon/evaluation_report.txt

# 查看可视化结果
eog runs/cascaded_eval_balloon/visualizations/14898532020_ba6199dd22_k.jpg

# 查看标签文件
cat runs/cascaded_eval_balloon/labels_two_stage/14898532020_ba6199dd22_k.txt
```

---

## 配置参数说明

### `run_cascaded_detection.sh` 核心参数

```bash
# YOLO模型路径
YOLO_MODEL="runs/detect/balloon_yolo11l_20251203_160322/weights/best.pt"

# 数据准备控制
FORCE_PREPARE=false     # 是否强制重新生成数据

# 第一阶段参数
STAGE1_CONF=0.05        # 低置信度，获取更多候选框
STAGE1_IOU=0.5          # 与GT匹配的IOU阈值

# 分类器参数
CLASSIFIER_TYPE="mobilenet"  # 分类器类型
INPUT_SIZE=112               # 输入尺寸
NUM_CLASSES=2                # 类别数

# 训练参数
BATCH_SIZE=32
EPOCHS=50
LR=0.001
```

### `run_cascaded_eval.sh` 核心参数

```bash
# 评估参数
SPLIT="val"                 # 评估集
STAGE1_CONF=0.05           # 第一阶段置信度
STAGE2_THRESHOLD=0.5       # 第二阶段阈值
YOLO_CONF=0.25             # 单阶段YOLO置信度（对比用）
```

---

## 目录结构总览

```
ultralytics/
├── data/                                    # 数据集目录（新增）
│   └── balloon_yolo11l_20251203_160322_cascaded_data_balloon/
│       ├── train/
│       │   ├── crops/                       # 裁剪的候选框
│       │   ├── data_list.json               # 数据列表
│       │   └── stats.json                   # 统计信息
│       └── val/
│           ├── crops/
│           ├── data_list.json
│           └── stats.json
│
├── runs/
│   ├── mobilenet/                           # 分类器权重（优化）
│   │   └── balloon_yolo11l_20251203_160322_20251209_124500/
│   │       ├── best.pt
│   │       ├── epoch_10.pt
│   │       └── ...
│   │
│   ├── cascaded_infer_balloon/              # 单张推理结果
│   │   └── balloon_yolo11l_20251203_160322/
│   │       └── image_cascaded.jpg
│   │
│   └── cascaded_eval_balloon/               # 批量评估结果（新增）
│       ├── visualizations/                  # 可视化对比图
│       ├── labels_single_stage/             # 单阶段标签
│       ├── labels_two_stage/                # 两阶段标签
│       ├── detailed_results.json            # 详细结果
│       └── evaluation_report.txt            # 评估报告
│
├── balloon_cascaded_detection.py            # 主程序（更新）
├── balloon_cascaded_infer_all.py            # 批量推理（更新）
├── run_cascaded_detection.sh                # 训练/推理脚本（更新）
├── run_cascaded_eval.sh                     # 批量评估脚本（新增）
├── CASCADED_DETECTION_GUIDE.md              # 使用指南
└── CASCADED_DETECTION_UPDATES.md            # 本文档
```

---

## 迁移到D1数据集

只需修改 `run_cascaded_detection.sh` 中的配置：

```bash
# YOLO模型路径（改为D1最好的模型）
YOLO_MODEL="path/to/D1_yolov8l_1280/weights/best.pt"

# 数据集配置
DATA_YAML="path/to/D1/data.yaml"

# 类别数（D1有3个前景类）
NUM_CLASSES=4  # 3个前景类 + 1个背景类

# 第一阶段置信度（D1是小目标，用更低阈值）
STAGE1_CONF=0.01

# IOU阈值（小目标定位困难，降低阈值）
STAGE1_IOU=0.3
```

数据集会自动保存到：
```
data/D1_yolov8l_1280_<timestamp>_cascaded_data_D1/
```

训练权重会自动保存到：
```
runs/mobilenet/D1_yolov8l_1280_<timestamp>_<training_time>/
```

---

## 常见问题

### Q1: 如何强制重新生成数据？
```bash
# 方法1：修改Shell脚本
FORCE_PREPARE=true bash run_cascaded_detection.sh prepare

# 方法2：直接使用Python
python balloon_cascaded_detection.py prepare \
    --yolo-model ... \
    --force
```

### Q2: 如何查看已保存的数据集统计？
```bash
cat data/balloon_yolo11l_20251203_160322_cascaded_data_balloon/train/stats.json
```

### Q3: 如何使用不同的YOLO模型？
只需修改 `YOLO_MODEL` 路径，所有相关路径会自动更新。

### Q4: 批量评估的可视化图像格式是什么？
左右对比图：
- 左侧：单阶段YOLO检测结果（绿色框）
- 右侧：两阶段级联检测结果（蓝色框）
- 顶部显示检测数量

### Q5: 如何快速对比两种方法的性能？
```bash
# 运行批量评估
bash run_cascaded_eval.sh

# 查看报告
cat runs/cascaded_eval_balloon/evaluation_report.txt
```

报告会显示：
- 平均数量准确率对比
- 提升最大的图像
- 下降最大的图像

---

## 性能优化建议

### 数据准备阶段
- **首次运行**: 约20-30秒（取决于图像数量）
- **后续运行**: 1秒（自动跳过）
- **建议**: 不同YOLO模型使用不同数据集

### 训练阶段
- **MobileNetV2**: 约20-30分钟（50 epochs, batch_size=32）
- **SimpleMLP**: 约10-15分钟（更轻量）
- **建议**: 使用MobileNetV2（有预训练权重，效果更好）

### 推理阶段
- **单阶段YOLO**: ~0.03秒/图
- **两阶段级联**: ~0.1秒/图（包含分类器）
- **批量评估**: ~1.3秒/图（包含可视化和标签保存）

---

## 总结

本次更新实现了4个关键功能，极大地提升了系统的易用性和可维护性：

1. ✅ **批量推理系统**: 完整的评估、可视化和标签保存功能
2. ✅ **智能路径管理**: 自动根据模型名称生成数据集和权重路径
3. ✅ **数据准备跳过**: 避免重复生成，节省时间
4. ✅ **清晰的组织结构**: 所有输出文件井然有序

现在可以更方便地：
- 在Balloon数据集上测试和验证
- 快速迁移到D1数据集
- 对比不同配置的性能
- 管理多个YOLO模型的级联检测结果

祝实验顺利，早日达到95%的目标准确率！🎯

