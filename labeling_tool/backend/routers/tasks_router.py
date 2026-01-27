from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
from datetime import datetime
import os

from .. import schemas, models, database
from ..inference_service import inference_service

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"]
)

def process_inference_task(task_id: int, request: schemas.InferenceRequest, db: Session):
    """
    后台任务：执行推理
    """
    try:
        task = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
        if not task:
            return

        # 更新状态为处理中
        task.status = "processing"
        
        # 加载模型
        inference_service.load_model(request.model_path, request.conf_thres, request.device)
        
        # 更新任务类别信息 (从加载的模型中获取)
        if inference_service.current_model:
            try:
                mapping = inference_service.current_model.category_mapping
                if mapping:
                    task.classes = [{"id": int(k), "name": v} for k, v in mapping.items()]
                    db.commit()
            except Exception as e:
                print(f"Failed to update task classes: {e}")

        # 解析图片源
        image_files = []
        if request.source_type == "folder":
            folder_path = Path(request.source_path)
            if folder_path.exists() and folder_path.is_dir():
                image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
                raw_files = []
                for ext in image_extensions:
                    raw_files.extend(folder_path.glob(f"*{ext}"))
                    raw_files.extend(folder_path.glob(f"*{ext.upper()}"))
                
                # 去重并排序 (解决 Windows 下大小写可能导致的重复问题)
                image_files = sorted(list(set(raw_files)))
        
        elif request.source_type == "list":
            # 假设是用逗号或分号分隔的路径字符串
            paths = request.source_path.replace(';', ',').split(',')
            for p_str in paths:
                p_str = p_str.strip()
                if not p_str: continue
                p = Path(p_str)
                if p.exists() and p.is_file():
                    image_files.append(p)

        # 更新任务总数
        task.total_images = len(image_files)
        task.processed_count = 0
        db.commit()

        # 开始处理
        for i, img_path in enumerate(image_files):
            try:
                # 记录图片信息
                from sahi.utils.cv import read_image
                img = read_image(str(img_path))
                h, w = img.shape[:2]

                # 创建图片记录
                db_image = models.ImageRecord(
                    task_id=task.id,
                    file_path=str(img_path),
                    file_name=img_path.name,
                    width=w,
                    height=h,
                    status="inferred"
                )
                db.add(db_image)
                db.flush() 

                # 执行推理
                detections = inference_service.infer_slice(
                    str(img_path),
                    slice_height=request.slice_height,
                    slice_width=request.slice_width,
                    overlap_height_ratio=request.overlap_height_ratio,
                    overlap_width_ratio=request.overlap_width_ratio,
                    postprocess_type=request.postprocess_type,
                    postprocess_match_metric=request.postprocess_metric,
                    postprocess_match_threshold=request.postprocess_threshold,
                    min_box_area=request.min_box_area,
                    max_detections=request.max_detections,
                    conf_thres=request.conf_thres,
                    cross_class_nms_enabled=request.cross_class_nms_enabled,
                    cross_class_nms_threshold=request.cross_class_nms_threshold
                )

                # 保存检测结果
                for det in detections:
                    db_det = models.Detection(
                        image_id=db_image.id,
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        x_center=det["x_center"],
                        y_center=det["y_center"],
                        width=det["width"],
                        height=det["height"],
                        is_manual=False
                    )
                    db.add(db_det)
                
                # 更新进度
                task.processed_count = i + 1
                db.commit()

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing image {img_path}: {e}")
                db.rollback() # 仅回滚当前图片的事务
                continue

        # 完成任务
        task.status = "completed"
        db.commit()
        
        # 注意：不再删除 _corrected.jpg 临时文件
        # 因为数据库中的 file_path 指向了这些文件，删除会导致前端无法预览
        # 这些文件现在保存在 runs/corrected_images/ 下，不会污染原数据集目录
        
    except Exception as e:
        print(f"Task failed: {e}")
        db.rollback() # 回滚整个任务事务
        
        # 尝试重新获取 task 对象来更新状态 (因为之前的 session 可能已经失效)
        try:
            task = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
            if task:
                task.status = "failed"
                db.commit()
        except Exception as inner_e:
             print(f"Failed to update task status: {inner_e}")

@router.post("/", response_model=schemas.InferenceTask)
def create_inference_task(
    request: schemas.InferenceRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """
    创建并启动一个新的推理任务
    """
    print(f"Creating task: source_path='{request.source_path}', type={request.source_type}")
    
    # 清理路径空白字符和可能的引号
    request.source_path = request.source_path.strip().strip('"').strip("'")
    
    # 打印调试信息
    print(f"DEBUG: Checking path: {repr(request.source_path)}", flush=True)

    # 简单验证
    if request.source_type == "folder":
        if not request.source_path:
            raise HTTPException(status_code=400, detail="Folder path is empty")
        if not Path(request.source_path).exists():
            print(f"Error: Folder not found: {repr(request.source_path)}", flush=True)
            raise HTTPException(status_code=404, detail=f"Folder not found: {request.source_path}")

    if request.source_type == "list":
        if not request.source_path:
            raise HTTPException(status_code=400, detail="File list is empty")
        # 简单验证列表中的第一个文件是否存在
        first_file = request.source_path.split(',')[0].strip()
        if not Path(first_file).exists():
             print(f"Error: First file in list not found: {repr(first_file)}", flush=True)
             raise HTTPException(status_code=404, detail=f"First file not found: {first_file}")

    # 创建任务记录
    # 对于 list 或 single 类型，folder_path 字段存 source_path 可能会很长，
    # 但 SQLite TEXT 没问题。或者存目录部分。这里直接存原始输入。
    task_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(request.model_path).stem}"
    
    # 尝试从模型中读取类别名称 (如果已经加载)
    classes = [{"id": 0, "name": "object"}]
    if inference_service.current_model and inference_service.current_model_path == request.model_path:
        try:
            # Sahi model category mapping
            # category_mapping = {'0': 'person', '1': 'bicycle', ...}
            mapping = inference_service.current_model.category_mapping
            if mapping:
                classes = [{"id": int(k), "name": v} for k, v in mapping.items()]
        except:
            pass

    db_task = models.InferenceTask(
        task_name=task_name,
        model_name=Path(request.model_path).name,
        folder_path=request.source_path[:255], # 截断显示
        status="pending",
        classes=classes
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # 启动后台任务
    background_tasks.add_task(process_inference_task, db_task.id, request, db)

    return db_task

@router.get("/", response_model=List[schemas.InferenceTask])
def list_tasks(db: Session = Depends(database.get_db)):
    """
    获取任务列表
    """
    return db.query(models.InferenceTask).order_by(models.InferenceTask.created_at.desc()).all()

@router.get("/{task_id}", response_model=schemas.InferenceTask)
def get_task(task_id: int, db: Session = Depends(database.get_db)):
    return db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()

@router.delete("/{task_id}")
def delete_task(task_id: int, db: Session = Depends(database.get_db)):
    """
    删除任务及其所有关联数据
    """
    task = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task)
    db.commit()
    return {"message": "Task deleted successfully"}
