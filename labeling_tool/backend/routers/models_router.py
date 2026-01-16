import os
import glob
from pathlib import Path
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(
    prefix="/models",
    tags=["models"]
)

class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    modified_time: str
    is_custom: bool = False

@router.get("/", response_model=List[ModelInfo])
def list_models():
    """
    扫描项目中的 .pt 模型文件
    优先扫描 runs/detect 下的训练权重
    """
    root_dir = Path(".")
    found_models = []
    seen_paths = set()

    # 1. 扫描训练结果目录 (重点)
    # runs/detect/exp/weights/best.pt
    training_patterns = [
        "runs/detect/**/best.pt", # 只查找 best.pt
    ]
    
    # 2. 扫描根目录预训练模型 (根据用户要求，暂时只关注 runs/detect)
    # 除非用户明确还需要根目录
    root_patterns = [] 

    # 先处理训练模型
    for pattern in training_patterns:
        for model_path in root_dir.glob(pattern):
            abs_path = str(model_path.resolve())
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            
            # 获取相对路径用于显示更友好的名称
            # 例如: runs/detect/train2/weights/best.pt
            # 显示名: train2/best.pt
            try:
                # 尝试获取相对于 runs/detect 的路径
                rel_parts = model_path.parts
                if "detect" in rel_parts:
                    idx = rel_parts.index("detect")
                    display_name = "/".join(rel_parts[idx+1:])
                else:
                    display_name = model_path.name
            except:
                display_name = model_path.name

            stat = model_path.stat()
            found_models.append({
                "name": f"[训练] {display_name}",
                "path": str(model_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": str(datetime.fromtimestamp(stat.st_mtime)),
                "is_custom": True
            })

    # 再处理根目录模型
    for pattern in root_patterns:
        for model_path in root_dir.glob(pattern):
            abs_path = str(model_path.resolve())
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            
            stat = model_path.stat()
            found_models.append({
                "name": f"[官方] {model_path.name}",
                "path": str(model_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": str(datetime.fromtimestamp(stat.st_mtime)),
                "is_custom": False
            })
    
    # 统一按修改时间倒序（最新的排最前）
    found_models.sort(key=lambda x: x["modified_time"], reverse=True)
    
    return found_models
