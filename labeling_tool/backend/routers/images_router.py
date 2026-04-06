from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from .. import schemas, models, database
from ..image_utils import process_perspective_correction, process_auto_stitch

router = APIRouter(
    prefix="/images",
    tags=["images"]
)

class CorrectionRequest(BaseModel):
    image_path: str
    points: List[List[float]] # [[x,y], [x,y], [x,y], [x,y]]

class StitchRequest(BaseModel):
    image_paths: List[str]

@router.post("/stitch")
def stitch_images(req: StitchRequest):
    try:
        if len(req.image_paths) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 images to stitch")
            
        new_path = process_auto_stitch(req.image_paths)
        return {"success": True, "new_path": new_path}
    except Exception as e:
        print(f"Stitching failed: {e}")
        # Return a 400 instead of 500 for expected algorithm failures so the frontend can handle it gracefully
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/correct-perspective")
def correct_perspective(req: CorrectionRequest):
    try:
        if len(req.points) != 4:
            raise HTTPException(status_code=400, detail="Need exactly 4 points")
            
        new_path = process_perspective_correction(req.image_path, req.points)
        return {"success": True, "new_path": new_path}
    except Exception as e:
        print(f"Correction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}", response_model=schemas.ImageRecord)
def get_image(image_id: int, db: Session = Depends(database.get_db)):
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@router.put("/{image_id}/settings")
def update_image_settings(image_id: int, settings: dict, db: Session = Depends(database.get_db)):
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    image.settings = settings
    db.commit()
    db.refresh(image)
    return {"success": True}

@router.post("/detections/", response_model=schemas.Detection)
def create_detection(det: schemas.DetectionCreate, db: Session = Depends(database.get_db)):
    """
    手动添加标注框 (通用接口)
    """
    if not det.image_id:
        raise HTTPException(status_code=400, detail="image_id is required")

    db_det = models.Detection(
        **det.dict()
    )
    db.add(db_det)
    
    # 更新图片状态
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == det.image_id).first()
    if image:
        image.status = "corrected"

    db.commit()
    db.refresh(db_det)
    return db_det

# 兼容旧接口
@router.post("/{image_id}/detections", response_model=schemas.Detection)
def add_detection(image_id: int, detection: schemas.DetectionCreate, db: Session = Depends(database.get_db)):
    detection.image_id = image_id
    return create_detection(detection, db)

@router.delete("/detections/{detection_id}")
def delete_detection(detection_id: int, db: Session = Depends(database.get_db)):
    """
    删除标注框
    """
    det = db.query(models.Detection).filter(models.Detection.id == detection_id).first()
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # 级联更新图片状态（可选）
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == det.image_id).first()
    if image:
        image.status = "corrected"
        
    db.delete(det)
    db.commit()
    return {"status": "success"}

@router.post("/save-base64")
def save_base64_image(req: dict):
    """
    保存前端传来的 Base64 图片数据
    """
    try:
        import base64
        import time
        from pathlib import Path
        
        image_data = req.get('image_data')
        prefix = req.get('prefix', 'image')
        
        if not image_data or not image_data.startswith('data:image/'):
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # 解析 base64
        header, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        
        # 确定扩展名
        ext = ".png"
        if "jpeg" in header or "jpg" in header:
            ext = ".jpg"
            
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        save_dir = project_root / "runs" / "stitched_images"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        new_filename = f"{prefix}_{timestamp}{ext}"
        new_path = save_dir / new_filename
        
        with open(new_path, "wb") as f:
            f.write(data)
            
        return {"success": True, "new_path": str(new_path)}
    except Exception as e:
        print(f"Save base64 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/detections/{detection_id}", response_model=schemas.Detection)
def update_detection(detection_id: int, det_update: schemas.DetectionCreate, db: Session = Depends(database.get_db)):
    db_det = db.query(models.Detection).filter(models.Detection.id == detection_id).first()
    if not db_det:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # 过滤掉 image_id，避免修改它
    update_data = det_update.dict(exclude_unset=True)
    if 'image_id' in update_data:
        del update_data['image_id']

    for key, value in update_data.items():
        setattr(db_det, key, value)
    
    # 更新图片状态
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == db_det.image_id).first()
    if image:
        image.status = "corrected"

    db.commit()
    db.refresh(db_det)
    return db_det

@router.post("/detections/", response_model=schemas.Detection)
def create_detection(det: schemas.DetectionCreate, db: Session = Depends(database.get_db)):
    """
    手动添加标注框 (通用接口)
    """
    if not det.image_id:
        raise HTTPException(status_code=400, detail="image_id is required")

    db_det = models.Detection(
        **det.dict()
    )
    db.add(db_det)
    
    # 更新图片状态
    image = db.query(models.ImageRecord).filter(models.ImageRecord.id == det.image_id).first()
    if image:
        image.status = "corrected"

    db.commit()
    db.refresh(db_det)
    return db_det
