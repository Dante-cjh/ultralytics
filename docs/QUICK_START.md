# Balloon 训练快速启动指南

## ✅ 检查完成

我已经完成了以下工作：

### 1. 训练脚本检查和修改 ✓

- ✅ **balloon_training.py**: 添加了命令行参数支持，包括 model, epochs, patience, device 等
- ✅ **balloon_training_with_slice.py**: 添加了 patience 参数，默认值30，可通过命令行配置
- ✅ **balloon_training_with_multi_slice.py**: 添加了 patience 参数，默认值30，可通过命令行配置
- ✅ **所有脚本都会自动生成 TensorBoard 日志**

### 2. Bash 自动化脚本 ✓

已生成三个bash脚本，用于批量训练不同模型：

| 脚本名称 | 训练方式 | 模型 | epochs | patience |
|---------|---------|------|--------|----------|
| `balloon_training_all_models.sh` | 普通训练 | yolo11m, yolo11l, yolo11x | 200 | 20 |
| `balloon_training_slice_all_models.sh` | 单尺度切片 | yolo11m, yolo11l, yolo11x | 200 | 20 |
| `balloon_training_multi_slice_all_models.sh` | 多尺度切片 | yolo11m, yolo11l, yolo11x | 200 | 20 |

### 3. 功能特性 ✓

- ✅ 自动化训练流程（依次训练3个模型）
- ✅ 训练完成后自动进行SAHI推理（验证集+测试集）
- ✅ 切片脚本第一次执行完整流程，后续仅训练（使用--train-only）
- ✅ 完善的日志记录和错误处理
- ✅ 统一的命名规则（包含时间戳）

---

## 🚀 如何运行

### 快速开始

```bash
# 1. 普通训练（推荐先运行这个）
bash balloon_training_all_models.sh

# 2. 单尺度切片训练
bash balloon_training_slice_all_models.sh

# 3. 多尺度切片训练
bash balloon_training_multi_slice_all_models.sh
```

### 后台运行

```bash
# 使用nohup后台运行
nohup bash balloon_training_all_models.sh > training_all_models.log 2>&1 &

# 查看运行状态
tail -f training_all_models.log
```

---

## 📊 查看日志

### TensorBoard（实时查看训练曲线）

```bash
# 启动TensorBoard
tensorboard --logdir runs/detect/

# 在浏览器打开
# http://localhost:6006
```

### 查看训练输出

```bash
# 查看实时日志（如果使用nohup）
tail -f training_all_models.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

---

## 📁 结果位置

### 训练结果

```
runs/detect/
├── balloon_yolo11m_{timestamp}/
│   ├── weights/
│   │   ├── best.pt          # ← 最佳模型
│   │   └── last.pt
│   ├── results.csv
│   └── results.png          # ← 训练曲线
├── balloon_yolo11l_{timestamp}/
└── balloon_yolo11x_{timestamp}/
```

### SAHI推理结果

```
runs/sahi_inference/
├── balloon_yolo11m_{timestamp}_val/    # 验证集推理
├── balloon_yolo11m_{timestamp}_test/   # 测试集推理
├── balloon_yolo11l_{timestamp}_val/
├── balloon_yolo11l_{timestamp}_test/
└── ...
```

---

## 🔧 自定义配置

### 修改GPU设备

编辑bash脚本中的 `DEVICE` 变量：

```bash
# 修改为GPU 0
DEVICE=0
```

### 修改训练参数

编辑bash脚本中的参数：

```bash
EPOCHS=300      # 增加训练轮数
BATCH=8         # 减小批次（如果GPU内存不足）
PATIENCE=30     # 增加早停耐心值
```

---

## 📖 详细文档

查看 **`BALLOON_TRAINING_GUIDE.md`** 获取完整文档，包括：

- 详细的参数说明
- TensorBoard使用指南
- 常见问题解答
- 性能优化建议
- 故障排除

---

## 📋 检查清单

运行前确认：

- [x] 数据集路径正确（默认：`/home/cjh/mmdetection/data/balloon/yolo_format`）
- [x] GPU 5 可用（`nvidia-smi` 查看）
- [x] 磁盘空间充足（建议 >50GB）
- [x] Python环境已激活
- [x] 依赖包已安装（ultralytics, sahi）

---

## 🎯 推荐运行顺序

```bash
# 第一步：普通训练（最快，作为baseline）
bash balloon_training_all_models.sh

# 第二步：单尺度切片训练
bash balloon_training_slice_all_models.sh

# 第三步：多尺度切片训练（最耗时，但效果可能最好）
bash balloon_training_multi_slice_all_models.sh

# 查看所有结果
tensorboard --logdir runs/detect/
```

---

## ⏱️ 预估时间

| 训练方式 | 单个模型 | 三个模型总计 |
|---------|---------|-------------|
| 普通训练 | ~2-3小时 | ~6-9小时 |
| 单尺度切片 | ~3-4小时 | ~9-12小时 |
| 多尺度切片 | ~4-6小时 | ~12-18小时 |

*实际时间取决于GPU性能、数据集大小和早停触发情况*

---

**一切准备就绪！运行脚本开始训练吧！🎉**

