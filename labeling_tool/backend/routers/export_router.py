from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
import shutil
import yaml
from datetime import datetime

from .. import schemas, models, database

router = APIRouter(
    prefix="/export",
    tags=["export"]
)

@router.post("/dataset")
def export_dataset(
    request: schemas.ExportRequest,
    db: Session = Depends(database.get_db)
):
    """
    导出选中任务的数据为 YOLO 格式数据集
    """
    # 验证任务是否存在
    tasks = db.query(models.InferenceTask).filter(
        models.InferenceTask.id.in_(request.task_ids)
    ).all()
    
    if not tasks:
        raise HTTPException(status_code=404, detail="未找到指定任务")
    
    # 创建导出目录
    output_path = Path(request.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建标准 YOLO 目录结构
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"
    
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图片
    total_images = 0
    total_labels = 0
    
    all_valid_images = []
    for task in tasks:
        for img_record in task.images:
            # 只导出已修订或已推理的图片
            if img_record.status not in ["inferred", "corrected"]:
                continue
            
            src_image_path = Path(img_record.file_path)
            if not src_image_path.exists():
                continue
            
            all_valid_images.append({
                "record": img_record,
                "task_id": task.id,
                "src_path": src_image_path
            })
    
    # 打乱并切分数据集
    import random
    # 简单的随机打乱
    random.shuffle(all_valid_images)
    
    split_index = int(len(all_valid_images) * request.split_ratio)
    train_set = all_valid_images[:split_index]
    val_set = all_valid_images[split_index:]
    
    def export_subset(subset, target_img_dir, target_lbl_dir):
        nonlocal total_images, total_labels
        for item in subset:
            task_id = item["task_id"]
            src_image_path = item["src_path"]
            img_record = item["record"]
            
            # 处理文件名冲突 (添加 task_id 前缀)
            if request.rename_files:
                new_filename = f"task_{task_id}_{src_image_path.name}"
            else:
                new_filename = src_image_path.name
                
            dst_image_path = target_img_dir / new_filename
            
            # 复制图片
            shutil.copy2(src_image_path, dst_image_path)
            
            # 生成标签文件
            label_path = target_lbl_dir / f"{Path(new_filename).stem}.txt"
            with open(label_path, 'w') as f:
                for det in img_record.detections:
                    # YOLO 格式：class_id x_center y_center width height
                    f.write(f"{det.class_id} {det.x_center:.6f} {det.y_center:.6f} {det.width:.6f} {det.height:.6f}\n")
                    total_labels += 1
            
            total_images += 1

    # 导出训练集
    export_subset(train_set, images_train, labels_train)
    # 导出验证集
    export_subset(val_set, images_val, labels_val)
    
    # 生成 data.yaml
    # 从数据库动态获取类别名称
    class_names = []
    # 尝试从第一个任务中获取类别信息（假设所有选中任务使用相同的类别体系）
    if tasks and tasks[0].classes:
        # tasks[0].classes 是一个列表 [{'id': 0, 'name': 'cls1'}, ...]
        # 我们需要按 id 排序并提取 name
        sorted_classes = sorted(tasks[0].classes, key=lambda x: x['id'])
        class_names = [c['name'] for c in sorted_classes]
    
    # 如果没有找到类别信息，尝试从数据库现有的 Detections 推断
    if not class_names:
        # 这里做一个简单的 fallback，或者默认为 object
        class_names = ["object"]
    
    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names)
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    return {
        "status": "success",
        "output_dir": str(output_path),
        "total_images": total_images,
        "total_labels": total_labels,
        "version_tag": request.version_tag
    }
