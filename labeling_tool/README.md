# Labeling Tool - 快速启动指南

## 系统简介
这是一个专为 Balloon/D1 等高分辨率数据集设计的**人机协作标注系统**。
核心特性：SAHI 切片推理 → 交互式修订 → 一键导出训练集。

---

## 一、环境准备

### 1. Python 依赖
确保已安装以下库（如果未安装，请运行下方命令）：

```bash
pip install fastapi uvicorn sqlalchemy pyyaml sahi
```

**说明**：
- `fastapi` + `uvicorn`: Web 服务框架
- `sqlalchemy`: 数据库 ORM
- `pyyaml`: 生成 data.yaml 配置文件
- `sahi`: 切片推理核心库

### 2. 前端依赖
前端使用 CDN 引入，**无需额外安装**。

---

## 二、启动系统

在项目根目录（ultralytics/）下运行：

```bash
python start_labeling_tool.py
```

启动成功后，控制台会显示：
```
🚀 Starting Labeling Tool...
📂 Backend: http://127.0.0.1:8000
💻 Frontend: Open labeling_tool/frontend/index.html in your browser
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

然后打开浏览器访问：**http://127.0.0.1:8000**

---

## 三、使用流程

### Step 1: 新建推理任务
1. 点击右上角 **"新建推理任务"**
2. 填写信息：
   - **图片文件夹**: 例如 `E:\data\balloon\images\val`
   - **模型权重**: 从下拉框选择（自动扫描 `runs/detect` 下的 `.pt` 文件）
   - **切片大小**: 默认 640（可根据图片分辨率调整）
   - **重叠率**: 默认 0.2（建议 0.15-0.3）
   - **置信度阈值**: 默认 0.25
3. 点击 **"开始推理"**，系统将在后台执行 SAHI 切片推理

### Step 2: 查看推理结果
1. 左侧 **"任务列表"** 会显示刚创建的任务
2. 点击任务卡片 → 进入 **"图片列表"**
3. 点击任意图片 → 中间画布加载大图和检测框

### Step 3: 修订标注
**操作说明：**
- **缩放**: 鼠标滚轮
- **平移**: 右键拖拽
- **选择框**: 左键点击红/绿色框
- **删除框**: 选中后按 `Delete` 键，或点击右侧列表的删除按钮
- **新建框**: 
  1. 先在右侧 **"当前标注类别"** 选择类别
  2. 在画布上拖拽创建矩形框
- **修改框**: 选中框后拖拽边角调整大小/位置
- **保存**: 修改会自动保存到数据库（也可点击右下角 "保存修改"）

**颜色说明：**
- 🟥 红色框：AI 推理生成
- 🟩 绿色框：人工修改/新增

### Step 4: 导出数据集
1. 返回 **"任务列表"**
2. 勾选需要导出的任务（可多选）
3. 点击右上角 **"导出选中"**
4. 填写导出路径（例如 `E:\datasets\balloon_v1.0`）和版本标签
5. 点击 **"开始导出"**

**导出结果：**
```
balloon_v1.0/
├── images/
│   ├── train/  (80% 图片)
│   └── val/    (20% 图片)
├── labels/
│   ├── train/  (对应的 .txt 标签)
│   └── val/
└── data.yaml   (YOLO 训练配置)
```

直接用于训练：
```bash
yolo detect train data=balloon_v1.0/data.yaml model=yolo11l.pt epochs=100
```

---

## 四、常见问题

### Q1: 模型列表为空？
**原因**: 系统扫描不到 `.pt` 文件。
**解决**: 
- 确保你的训练权重在 `runs/detect/**/weights/*.pt` 路径下
- 或者把模型文件放到项目根目录

### Q2: 推理任务一直显示 "processing"？
**原因**: 后台任务可能报错。
**解决**: 查看控制台日志，通常是模型路径错误或图片文件夹不存在。

### Q3: 画布加载不出图片？
**原因**: 文件路径中有特殊字符或中文导致编码问题。
**解决**: 
- 检查图片路径是否存在
- 尽量使用英文路径

### Q4: 如何添加更多类别？
修改 `labeling_tool/frontend/index.html` 第 249 行：
```javascript
const classList = ref([
    { id: 0, name: 'balloon' },
    { id: 1, name: 'person' },  // 新增
    { id: 2, name: 'car' },     // 新增
]);
```

---

## 五、关闭系统

在控制台按 `Ctrl+C` 即可停止服务。

数据库文件 `labeling_tool/data.db` 会保留所有标注记录，下次启动时自动加载。

---

**技术支持**: 如有问题，请查看控制台错误日志或联系开发者。
