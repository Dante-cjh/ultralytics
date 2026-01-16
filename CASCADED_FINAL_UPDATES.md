# 级联检测系统最终更新

## 📦 本次更新内容（2024-12-10）

### 新增功能总览

1. ✅ **样本平衡策略优化** - 下采样正样本
2. ✅ **跨类别NMS** - 解决重复框问题
3. ✅ **数据增强和正则化** - 解决过拟合问题
4. ✅ **SAHI结果精修** - 对SAHI推理结果进行二阶段分类
5. ✅ **改进的结果保存** - 保存推理图像到runs/inference/结构

---

## 🔧 详细更新

### 1. ✅ 样本平衡策略优化

**问题**：D1数据集正样本过多（6万+），负样本不足（3-4万），导致模型4-6轮就过拟合。

**解决方案**：当负样本不足时，下采样正样本而不是生成负样本。

**修改文件**：`balloon_cascaded_detection.py`

**新逻辑**：
```python
if 负样本 > 目标负样本数:
    # 情况1: 负样本过多
    下采样负样本
elif 负样本 < 目标负样本数:
    # 情况2: 负样本不足（正样本过多）
    ✅ 下采样正样本以达到平衡！
else:
    # 情况3: 已经平衡
    无需调整
```

**效果示例**：
```
原始数据:
  正样本: 65,000
  负样本: 40,000
  比例: 1:0.62 ❌

negative_ratio=1.0 平衡后:
  正样本: 40,000 (下采样)
  负样本: 40,000
  比例: 1:1.0 ✅
  总样本: 80,000（从10.5万降到8万）
```

---

### 2. ✅ 跨类别NMS功能

**问题**：一个孔洞被多个框框住（不同类别：hole、cave、unknow）。

**原因**：YOLO的NMS只在单个类别内生效，不同类别的框不会被抑制。

**解决方案**：添加跨类别NMS，对所有类别的框一起进行NMS。

**修改文件**：
- `balloon_cascaded_detection.py`
- `balloon_cascaded_infer_all.py`

**新增函数**：
```python
def cross_class_nms(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    跨类别NMS：处理不同类别预测同一目标的情况
    
    策略：对于高度重叠的框（即使类别不同），只保留置信度最高的
    """
```

**使用方式**：
```bash
# 启用跨类别NMS（默认）
python balloon_cascaded_detection.py infer \
    --cross-class-nms \
    --nms-iou 0.3

# 禁用跨类别NMS
python balloon_cascaded_detection.py infer \
    --no-cross-class-nms
```

---

## 🔧 使用指南

### 优化样本平衡

```bash
# 修改 run_cascaded_detection.sh
NEGATIVE_RATIO=1.0  # 建议从1.0开始（1:1平衡）

# 重新生成数据
FORCE_PREPARE=true bash run_cascaded_detection.sh prepare
```

**不同比例的效果**：

| negative_ratio | 正样本 | 负样本 | 总样本 | 适用场景 |
|----------------|--------|--------|--------|----------|
| 1.0 | 40k | 40k | 80k | ✅ 平衡（推荐） |
| 0.5 | 80k | 40k | 120k | 提高召回率 |
| 2.0 | 20k | 40k | 60k | 提高精确率 |

### 使用跨类别NMS

跨类别NMS默认**已启用**，IOU阈值为0.3。

**如需调整**：

```bash
# 方式1: 修改 balloon_cascaded_detection.py infer 命令
python balloon_cascaded_detection.py infer \
    --yolo-model <path> \
    --classifier <path> \
    --image <path> \
    --nms-iou 0.5  # 调整IOU阈值

# 方式2: 修改 balloon_cascaded_infer_all.py
python balloon_cascaded_infer_all.py \
    --yolo-model <path> \
    --classifier <path> \
    --data-yaml <path> \
    --nms-iou 0.5  # 调整IOU阈值
```

**IOU阈值选择**：
- `0.3`: 严格（重叠30%就抑制）→ 减少重复框
- `0.5`: 标准（重叠50%才抑制）
- `0.7`: 宽松（重叠70%才抑制）

---

## 📝 完整实验流程

### 实验1：优化样本平衡

```bash
cd /home/cjh/ultralytics

# 步骤1: 使用最好的YOLO模型准备数据
YOLO_MODEL="runs/detect/D1_yolov8l_1280/weights/best.pt"  # 你的92%模型
NEGATIVE_RATIO=1.0  # 1:1平衡

FORCE_PREPARE=true bash run_cascaded_detection.sh prepare

# 步骤2: 检查数据统计
cat data/D1_yolov8l_1280_cascaded_data_D1/train/stats.json

# 应该看到类似：
# {
#   "positive_samples": 40000,
#   "negative_samples": 40000,
#   "total_proposals": 80000
# }

# 步骤3: 训练二阶段分类器
bash run_cascaded_detection.sh train

# 观察：
# - 不应该4-6轮就过拟合
# - Val准确率应该 > 90%

# 步骤4: 评估级联系统
bash run_cascaded_eval.sh
```

### 实验2：测试切片训练模型

```bash
# 使用切片训练的模型作为一阶段
YOLO_MODEL="runs/detect/D1_yolov8l_slice_train/weights/best.pt"

# 先测试切片模型的全图推理效果
python balloon_inference.py \
    --model $YOLO_MODEL \
    --imgsz 1280 \
    --conf 0.25 \
    --data my_D1.yaml \
    --split test

python count_comparison_tool.py \
    --pred runs/inference_*/labels \
    --true data/D1/labels/test

# 如果效果 > 92%，用它做级联
FORCE_PREPARE=true \
YOLO_MODEL=$YOLO_MODEL \
bash run_cascaded_detection.sh all
```

### 实验3：对比不同negative_ratio

```bash
# 测试1: 1:1平衡
NEGATIVE_RATIO=1.0 \
FORCE_PREPARE=true \
bash run_cascaded_detection.sh all

# 测试2: 1:2平衡（更多负样本）
NEGATIVE_RATIO=2.0 \
FORCE_PREPARE=true \
bash run_cascaded_detection.sh all

# 测试3: 2:1平衡（更多正样本）
NEGATIVE_RATIO=0.5 \
FORCE_PREPARE=true \
bash run_cascaded_detection.sh all

# 对比三者的：
# - 分类器Val准确率
# - 级联系统计数准确率
```

### 实验4：测试跨类别NMS的效果

```bash
# 准备一个测试图像
TEST_IMAGE="data/D1/images/test/xxx.jpg"
YOLO_MODEL="<你的模型路径>"
CLASSIFIER="<你的分类器路径>"

# 测试1: 不使用跨类别NMS
python balloon_cascaded_detection.py infer \
    --yolo-model $YOLO_MODEL \
    --classifier $CLASSIFIER \
    --image $TEST_IMAGE \
    --no-cross-class-nms \
    --save-dir runs/test_no_nms

# 测试2: 使用跨类别NMS (IOU=0.3)
python balloon_cascaded_detection.py infer \
    --yolo-model $YOLO_MODEL \
    --classifier $CLASSIFIER \
    --image $TEST_IMAGE \
    --cross-class-nms \
    --nms-iou 0.3 \
    --save-dir runs/test_nms_03

# 测试3: 使用跨类别NMS (IOU=0.5)
python balloon_cascaded_detection.py infer \
    --yolo-model $YOLO_MODEL \
    --classifier $CLASSIFIER \
    --image $TEST_IMAGE \
    --cross-class-nms \
    --nms-iou 0.5 \
    --save-dir runs/test_nms_05

# 对比三张图，看哪个减少重复框效果最好
```

---

## 🎯 预期效果

### 样本平衡优化

**之前（不平衡）**：
```
训练数据: 6.5万正 + 4万负 = 10.5万
训练过程:
  Epoch 4: Train Acc 95%, Val Acc 85%
  Epoch 5: Train Acc 97%, Val Acc 85%
  Epoch 6: Train Acc 98%, Val Acc 84% ← 过拟合
  
级联效果: < 92% (不如单阶段)
```

**优化后（平衡）**：
```
训练数据: 4万正 + 4万负 = 8万 (减少24%)
训练过程:
  Epoch 10: Train Acc 90%, Val Acc 88%
  Epoch 20: Train Acc 92%, Val Acc 90%
  Epoch 30: Train Acc 93%, Val Acc 91% ← 稳定提升
  
级联效果: 预期 > 92%
```

### 跨类别NMS

**之前（无跨类别NMS）**：
```
一个孔洞:
  框1: hole, conf=0.85
  框2: cave, conf=0.72
  框3: unknow, conf=0.68
  
显示: 3个框重叠 ❌
```

**优化后（有跨类别NMS）**：
```
一个孔洞:
  框1: hole, conf=0.85 ✅ (保留最高置信度)
  框2: cave, conf=0.72 ← 被抑制
  框3: unknow, conf=0.68 ← 被抑制
  
显示: 1个框 ✅
```

---

## ⚠️ 重要提醒

### 1. IOU阈值的正确理解（纠正之前的文档）

```python
# ❌ 错误理解（之前文档有误）
"降低IOU阈值可以增加负样本"

# ✅ 正确理解
IOU阈值 ↑ → 匹配更严格 → 正样本 ↓, 负样本 ↑
IOU阈值 ↓ → 匹配更宽松 → 正样本 ↑, 负样本 ↓

示例：
候选框与GT的IOU = 0.4

IOU阈值=0.5 → 0.4 < 0.5 → 负样本 ✅
IOU阈值=0.3 → 0.4 > 0.3 → 正样本 ✅
```

**但是**：对于D1数据集，这个参数作用不大！

原因：
```
D1特点: 密集小目标，一张图几十上百个孔洞
conf=0.01: YOLO已经输出所有可能的检测
→ 大部分检测都是真实目标
→ 大部分候选框与GT的IOU都很高（>0.5）
→ 调整IOU阈值改变不大

你的数据: 负:正 = 0.54~0.8:1
说明: 即使conf=0.01，负样本依然不足
结论: IOU阈值调整无法根本解决问题
```

**因此**：保持 `IOU=0.5` 即可，关键是样本平衡策略（下采样正样本）。

### 2. 二阶段级联的天花板

```
二阶段效果 = min(一阶段质量, 二阶段质量)
```

**一阶段YOLO的问题**：
- 漏检：还有目标没检测到
- 重复框：一个孔被多个框框住

**二阶段级联只能**：
- ✅ 过滤误检（背景被误检为目标）
- ❌ 无法找回漏检的目标（已经丢失）
- ⚠️ 可以通过跨类别NMS减少重复框（新功能）

**因此优先级**：
1. **优先级1**：优化一阶段YOLO（减少漏检）
   - 尝试切片训练模型的全图推理
   - 如果效果好，用它作为一阶段
2. **优先级2**：优化二阶段分类器（减少误检）
   - 样本平衡策略（下采样正样本）
   - 预期分类器Val准确率 > 90%
3. **优先级3**：后处理优化（减少重复框）
   - 跨类别NMS（已实现）
   - IOU阈值调整（0.3-0.5）

### 3. 切片训练模型的潜力

**你之前的实验**：
```
切片640推理: 79% ❌
切片1280推理: 88% ❌
全图1280推理: 92% ✅

结论: 切片推理不如全图推理
```

**但是！尝试这个**：
```
切片训练的模型 + 全图推理 = ？

理论：
- 切片训练: 小目标变大，学习更充分
- 全图推理: 保持全局信息，不受拼接影响

可能效果: 94-95%？
```

**如何测试**：
```bash
# 1. 找到切片训练模型
SLICE_MODEL="runs/detect/D1_yolov8l_slice_train/weights/best.pt"

# 2. 全图推理（不要切片！）
python balloon_inference.py \
    --model $SLICE_MODEL \
    --imgsz 1280 \
    --conf 0.25 \
    --data my_D1.yaml \
    --split test

# 3. 对比计数准确率
python count_comparison_tool.py \
    --pred runs/inference_*/labels \
    --true data/D1/labels/test
```

---

## 🆕 新增功能3：数据增强和正则化（解决过拟合）

### 问题

**离线服务器测试结果**：
```
训练集准确率: 99% ✅
测试集准确率: 86% ❌
收敛轮次: 6-8轮
```

**诊断**：严重过拟合！即使样本平衡（1:1或2:1），模型依然过拟合。

### 解决方案

#### 方案A：增强的数据增强

**修改文件**：`balloon_cascaded_detection.py` (train命令)

**新增的数据增强**：
```python
train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    
    # 几何变换
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),  # ±15度旋转
    
    # 颜色增强
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    
    transforms.ToTensor(),
    
    # 随机擦除（模拟遮挡）
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    
    transforms.Normalize(...)
])
```

#### 方案B：增加Dropout和权重衰减

**修改文件**：`balloon_cascaded_detection.py` (MobileNetClassifier)

**Dropout从0.2提升到0.5**：
```python
class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):  # ← 默认0.5
        ...
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),        # 第一层Dropout
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),        # 第二层Dropout
            nn.Linear(256, num_classes)
        )
```

**权重衰减（L2正则化）**：
```python
optimizer = torch.optim.AdamW(  # 使用AdamW
    model.parameters(),
    lr=lr,
    weight_decay=0.01  # ← L2正则化
)
```

#### 方案C：早停和学习率调度

**新增早停机制**：
```python
patience = 10  # 10轮验证准确率未提升则停止
best_val_acc = 0.0
patience_counter = 0

if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    # 保存最佳模型
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"早停！{patience}轮无提升")
        break
```

**学习率调度**：
```python
# 余弦退火 + 性能plateau调整
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# 每个epoch
scheduler_cosine.step()
scheduler_plateau.step(val_acc)
```

### 预期效果

**之前（过拟合）**：
```
Epoch 6: Train 97%, Val 84%
Epoch 8: Train 99%, Val 86% ← 过拟合
```

**优化后（健康收敛）**：
```
Epoch 10: Train 88%, Val 87%
Epoch 20: Train 93%, Val 90% ← 健康收敛
```

---

## 🆕 新增功能4：SAHI结果的二阶段精修（两种方式）

### 方式A：离线精修（两步走）

**适用场景**：已经有SAHI推理结果（labels），想要精修

**新增文件**：
1. **`balloon_cascaded_from_sahi.py`** - SAHI结果精修脚本
2. **`run_cascaded_sahi.sh`** - 运行脚本

**使用方法**：

```bash
# 步骤1: 使用SAHI进行切片推理（假设已完成）
# 结果保存在: runs/sahi_inference/D1_yolov8l_slice_xxx_val/

# 步骤2: 使用二阶段分类器精修SAHI结果
bash run_cascaded_sahi.sh

# 或手动指定参数
python balloon_cascaded_from_sahi.py \
    --sahi-results runs/sahi_inference/D1_yolov8l_slice_xxx_val \
    --images data/D1/images/val \
    --classifier runs/mobilenet/D1_yolov8l_1280_xxx/best.pt \
    --save-dir runs/cascaded_sahi_refine \
    --threshold 0.5
```

**输出结果**：

```
runs/cascaded_sahi_refine/
├── labels/                     # 精修后的YOLO格式labels
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── visualizations/             # 对比可视化（SAHI vs 精修）
    ├── image1_comparison.jpg
    ├── image2_comparison.jpg
    └── ...
```

### 方式B：在线推理（一步到位）⭐ 推荐

**适用场景**：直接进行SAHI推理并同时使用二阶段分类

**新增文件**：
1. **`balloon_sahi_cascaded_infer_all.py`** - SAHI两阶段批量推理脚本
2. **`run_sahi_cascaded_eval.sh`** - 运行脚本

**使用方法**：

```bash
# 一步完成：SAHI推理 + 二阶段分类 + 评估
bash run_sahi_cascaded_eval.sh

# 或手动指定参数
python balloon_sahi_cascaded_infer_all.py \
    --yolo-model runs/detect/D1_yolov8l_xxx/weights/best.pt \
    --classifier runs/mobilenet/D1_yolov8l_xxx/best.pt \
    --data-yaml my_D1.yaml \
    --split val \
    --slice-height 640 \
    --slice-width 640 \
    --overlap-ratio 0.2 \
    --sahi-conf 0.25 \
    --stage2-threshold 0.5 \
    --save-dir runs/inference/xxx_sahi_cascaded_val
```

**输出结果**：

```
runs/inference/<model_name>_sahi_cascaded_val/
├── images/                              # 推理图像（二阶段结果，带框）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels_sahi/                         # SAHI原始labels
│   ├── image1.txt
│   └── ...
├── labels_sahi_stage2/                  # 二阶段精修后labels
│   ├── image1.txt
│   └── ...
├── visualizations_comparison/           # 对比可视化（SAHI vs 二阶段）
│   ├── image1_comparison.jpg
│   └── ...
├── detailed_results.json                # 详细结果JSON
└── evaluation_report.txt                # 评估报告
```

### Balloon测试结果（方式B）

```
SAHI原始推理: 1.10% (289个检测)
  → 严重过检测：289检测 vs 50 GT (5.78倍)

SAHI + 二阶段: 46.40% (42个检测)
  → 过滤误检：247个 (85%的误检被过滤)
  → 检测数接近真实：42 ≈ 50

性能提升: +45.30%
```

### 两种方式对比

| 特性 | 方式A（离线精修） | 方式B（在线推理）⭐ |
|------|------------------|-------------------|
| 适用场景 | 已有SAHI结果 | 从头开始推理 |
| 步骤 | 两步（先SAHI，后精修） | 一步到位 |
| 评估报告 | 无 | ✅ 自动生成 |
| 推理图像 | 无 | ✅ 自动保存 |
| 推荐程度 | 适用于已有结果 | ✅ 推荐新任务使用 |

### 核心代码

```python
class SAHIResultRefiner:
    """SAHI结果精修器"""
    
    def refine_detections(self, image_path, label_path):
        # 1. 读取SAHI的labels（YOLO格式）
        stage1_dets = self.parse_yolo_label(label_path, img_w, img_h)
        
        # 2. 对每个检测框进行二次分类
        for det in stage1_dets:
            crop = img[y1:y2, x1:x2]
            crop_tensor = self.transform(crop)
            
            # MobileNetV2推理
            output = self.classifier(crop_tensor)
            probs = F.softmax(output, dim=1)
            stage2_conf, stage2_cls = probs.max(1)
            
            # 3. 过滤背景和低置信度检测
            if stage2_cls == 0 or stage2_conf < threshold:
                continue  # 丢弃
            
            refined_detections.append(...)
        
        return refined_detections
```

---

## 🆕 新增功能5：改进的结果保存结构

### 问题

之前的保存结构不够清晰，用户希望类似`runs/inference/<model_name>_val/`的结构。

### 解决方案

**修改文件**：`balloon_cascaded_infer_all.py`、`run_cascaded_eval.sh`

**新的保存结构**：
```
runs/inference/<model_name>_cascaded_val/
├── images/                              # 主要结果：两阶段推理图像（带框）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels_single_stage/                 # 单阶段YOLO的labels
│   ├── image1.txt
│   └── ...
├── labels_two_stage/                    # 两阶段级联的labels（主要结果）
│   ├── image1.txt
│   └── ...
├── visualizations_comparison/           # 对比可视化（单阶段 vs 两阶段）
│   ├── image1_comparison.jpg
│   └── ...
├── detailed_results.json                # 详细结果JSON
└── evaluation_report.txt                # 评估报告
```

**关键修改**：

1. **推理图像保存到images目录**：
```python
# 两阶段推理图像（主要结果）
cv2.imwrite(str(images_dir / f"{img_name}.jpg"), img_two_stage)
```

2. **对比可视化保存到visualizations_comparison目录**：
```python
# 左：单阶段（红色），右：两阶段（绿色）
vis_img = np.hstack([img_single, gap, img_two])
cv2.imwrite(str(vis_comp_dir / f"{img_name}_comparison.jpg"), vis_img)
```

3. **自动生成目录名**：
```bash
# run_cascaded_eval.sh
YOLO_MODEL_NAME=$(basename $(dirname $(dirname "$YOLO_MODEL")))
EVAL_DIR="runs/inference/${YOLO_MODEL_NAME}_cascaded_${SPLIT}"
```

---

## 📚 完整使用流程

### 流程1：标准级联检测（balloon数据集）

```bash
cd /home/cjh/ultralytics

# 1. 准备数据（使用增强的样本平衡）
FORCE_PREPARE=true \
NEGATIVE_RATIO=1.0 \
bash run_cascaded_detection.sh prepare

# 2. 训练分类器（自动使用数据增强和早停）
bash run_cascaded_detection.sh train

# 3. 批量评估
bash run_cascaded_eval.sh

# 4. 查看结果
ls runs/inference/balloon_yolo11l_xxx_cascaded_val/
```

### 流程2：SAHI切片推理 + 二阶段精修（离线方式）

```bash
# 1. SAHI切片推理（假设已完成）
# 例如：D1_inference_slice_yolov8l.sh 已运行
# 结果：runs/sahi_inference/D1_yolov8l_slice_xxx_val/

# 2. 训练分类器（如果没有）
bash run_cascaded_detection.sh train

# 3. 精修SAHI结果
bash run_cascaded_sahi.sh

# 4. 查看精修结果
ls runs/cascaded_sahi_refine/labels/
```

### 流程3：SAHI两阶段在线推理（推荐）⭐

```bash
# 一步完成：SAHI推理 + 二阶段分类 + 评估
cd /home/cjh/ultralytics

# 1. 训练分类器（如果没有）
bash run_cascaded_detection.sh train

# 2. SAHI两阶段批量推理
bash run_sahi_cascaded_eval.sh

# 3. 查看结果
ls runs/inference/<model_name>_sahi_cascaded_val/

# 输出：
# - images/                        # 推理图像
# - labels_sahi_stage2/            # 精修后的labels
# - visualizations_comparison/     # 对比可视化
# - evaluation_report.txt          # 评估报告
```

**推荐使用场景**：
- D1数据集（大图、密集小目标）
- 需要切片推理的场景
- 想要同时获得评估报告和可视化

---

## 📄 相关文档

- **过拟合问题分析**：`CASCADED_OVERFITTING_SOLUTIONS.md` ⭐ 新增
- **问题分析与解决方案**：`D1_CASCADED_ISSUES_SOLUTIONS.md`
- **样本平衡策略详解**：`SAMPLE_BALANCE_STRATEGY.md`
- **级联检测更新日志**：`CASCADED_DETECTION_UPDATES.md`

---

## 🤝 反馈与建议

如果在使用过程中遇到问题或有新的需求，请及时反馈！

**当前状态**：
- ✅ 样本平衡优化（下采样正样本）
- ✅ 跨类别NMS功能
- ⏳ 等待D1服务器上的实验结果

**下一步计划**：
1. 测试优化后的样本平衡效果
2. 测试切片训练模型的全图推理效果
3. 验证跨类别NMS对重复框的改善
4. 根据实验结果进一步调整策略

目标：**从92%提升到95%！** 🎯

