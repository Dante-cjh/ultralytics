from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

# Detection Schemas
class DetectionBase(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    is_manual: bool = False

class DetectionCreate(DetectionBase):
    image_id: Optional[int] = None # 创建时需要

class Detection(DetectionBase):
    id: int
    image_id: int

    class Config:
        from_attributes = True

# Image Schemas
class ImageRecordBase(BaseModel):
    file_path: str
    file_name: str
    width: int
    height: int
    status: str

class ImageRecord(ImageRecordBase):
    id: int
    task_id: int
    detections: List[Detection] = []

    class Config:
        from_attributes = True

# Task Schemas
class InferenceTaskBase(BaseModel):
    task_name: str
    model_name: str
    folder_path: str

class InferenceTask(InferenceTaskBase):
    id: int
    created_at: datetime
    status: str
    total_images: int = 0
    processed_count: int = 0
    classes: List[dict] = []
    images: List[ImageRecord] = []

    class Config:
        from_attributes = True

# Request Schemas
class ScanRequest(BaseModel):
    folder_path: str

class InferenceRequest(BaseModel):
    # 核心路径参数
    source_type: str = "folder" # "folder", "list", "single"
    source_path: str # 文件夹路径 或 单张图片路径 或 多张图片路径(逗号分隔)
    model_path: str
    
    # 基础参数
    conf_thres: float = 0.3
    device: str = "cuda:0"
    
    # 切片参数
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.15
    overlap_width_ratio: float = 0.15
    
    # 后处理参数
    postprocess_type: str = "NMS" # NMS, NMM
    postprocess_threshold: float = 0.6
    postprocess_metric: str = "IOS" # IOS, IOU
    
    # 过滤参数
    min_box_area: int = 200
    max_detections: int = 50000
    
    # v3新增
    cross_class_nms_enabled: bool = True
    cross_class_nms_threshold: float = 0.5

class ExportRequest(BaseModel):
    task_ids: List[int]
    output_dir: str
    version_tag: str
    split_ratio: float = 0.8  # 训练集比例 (0.0 - 1.0)
    rename_files: bool = True # 是否重命名文件以避免冲突