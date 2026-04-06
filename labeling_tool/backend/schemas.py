from typing import List, Optional
from pydantic import BaseModel, Field, validator
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
    settings: Optional[dict] = Field(default_factory=dict)

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


# 弹片 / 弹道分析（预览模式「计算」）
class BallisticAnalysisRequest(BaseModel):
    """centers_px 与 pixel_width/height 为同一像素坐标系（如检测框中心）。"""

    pixel_width: float = Field(gt=0, description="图像宽度（像素）")
    pixel_height: float = Field(gt=0, description="图像高度（像素）")
    real_width_m: float = Field(gt=0, description="实际宽度（米）")
    real_height_m: float = Field(gt=0, description="实际高度（米）")
    distance_m: float = Field(gt=0, description="拍摄距离（米）")
    is_vertical: bool = Field(True, description="True=立式，False=卧式")
    baseline_px: float = Field(0.0, description="基准线位置（像素）；≤0 时用图像中心")
    centers_px: List[List[float]] = Field(..., min_items=1, description="[[cx, cy], ...] 像素坐标")
    hole_areas_px: Optional[List[float]] = Field(None, description="与 centers 顺序一致的弹孔面积（像素²）")

    @validator("centers_px")
    def _validate_centers(cls, v: List[List[float]]) -> List[List[float]]:
        for i, p in enumerate(v):
            if len(p) != 2:
                raise ValueError(f"centers_px[{i}] 必须为 [cx, cy]")
        return v


class BallisticAnalysisResponse(BaseModel):
    count: int
    dispersion_angle_deg: float
    azimuth_angle_deg: float
    fragment_density_per_m2: float
    max_hole_area_px: Optional[float] = None
    min_hole_area_px: Optional[float] = None