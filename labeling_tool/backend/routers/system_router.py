import subprocess
import sys
import os
import shutil
import time
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from pathlib import Path

router = APIRouter(
    prefix="/system",
    tags=["system"]
)

# 项目根目录（用于上传文件存储）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class SystemStatus(BaseModel):
    cuda_available: bool
    device_name: str
    device_count: int
    torch_version: str

class PathResponse(BaseModel):
    path: str
    
class PathsResponse(BaseModel):
    paths: List[str]

class ScanFolderRequest(BaseModel):
    path: str

@router.post("/scan-folder", response_model=PathsResponse)
def scan_folder(req: ScanFolderRequest):
    folder_path = Path(req.path)
    if not folder_path.exists() or not folder_path.is_dir():
        return {"paths": []}
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    raw_files = []
    for ext in image_extensions:
        raw_files.extend(folder_path.glob(f"*{ext}"))
        raw_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # 去重并排序
    image_files = sorted(list(set([str(p) for p in raw_files])))
    return {"paths": image_files}

@router.get("/status", response_model=SystemStatus)
def get_system_status():
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except:
            device_name = "CUDA (Unknown)"
    
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "torch_version": torch.__version__
    }

def run_dialog_script(mode):
    """
    运行独立的 Python 脚本来显示弹窗
    这样可以避免多进程的开销，且独立环境启动更快
    """
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "file_dialogs.py")
    try:
        # 使用当前的 python 解释器
        python_exe = sys.executable
        
        # 临时移除 STARTUPINFO 以解决部分环境下弹窗失败的问题
        startupinfo = None
        # if os.name == 'nt':
        #     startupinfo = subprocess.STARTUPINFO()
        #     startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(
            [python_exe, script_path, mode],
            capture_output=True,
            text=True,
            encoding='utf-8',
            startupinfo=startupinfo
        )
        
        if result.returncode != 0:
            print(f"Dialog script failed: {result.stderr}")
            
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running dialog script: {e}")
        return ""

@router.get("/pick-folder", response_model=PathResponse)
def pick_folder():
    path = run_dialog_script('folder')
    return {"path": path}

@router.get("/pick-file", response_model=PathResponse)
def pick_file():
    path = run_dialog_script('file')
    return {"path": path}

@router.get("/pick-files", response_model=PathsResponse)
def pick_files():
    output = run_dialog_script('files')
    paths = []
    if output:
        paths = output.split('|')
    return {"paths": paths}


# ── 服务器文件浏览器 ──────────────────────────────────────────────

class DirItem(BaseModel):
    name: str
    path: str
    type: str           # "dir" | "file"
    size_mb: Optional[float] = None

class BrowseDirResponse(BaseModel):
    current: str
    parent: Optional[str]
    items: List[DirItem]

@router.get("/browse-dir", response_model=BrowseDirResponse)
def browse_dir(path: str = Query(default="")):
    """
    浏览服务器目录，返回子目录和模型权重文件列表。
    path 为空时默认返回项目根目录。
    """
    MODEL_EXTS = {".pt", ".pth", ".onnx"}

    if not path:
        target = _PROJECT_ROOT
    else:
        target = Path(path)

    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {path}")

    items: List[DirItem] = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if item.is_dir():
                # 跳过隐藏目录和常见无关目录
                if item.name.startswith('.') or item.name in {"__pycache__", "node_modules", ".git"}:
                    continue
                items.append(DirItem(name=item.name, path=str(item), type="dir"))
            elif item.suffix.lower() in MODEL_EXTS:
                size_mb = round(item.stat().st_size / 1024 / 1024, 1)
                items.append(DirItem(name=item.name, path=str(item), type="file", size_mb=size_mb))
    except PermissionError:
        pass

    parent = str(target.parent) if target.parent != target else None

    return BrowseDirResponse(current=str(target), parent=parent, items=items)


# ── 本地图片上传 ──────────────────────────────────────────────────

class UploadResponse(BaseModel):
    paths: List[str]

@router.post("/upload-images", response_model=UploadResponse)
async def upload_images(files: List[UploadFile] = File(...)):
    """
    接收用户从本地浏览器上传的图片，保存到服务器临时目录，返回服务器端路径。
    """
    ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    upload_dir = _PROJECT_ROOT / "runs" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        # 用时间戳避免文件名冲突
        ts = int(time.time() * 1000)
        safe_name = f"{ts}_{Path(file.filename).name}"
        save_path = upload_dir / safe_name
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(save_path))

    return UploadResponse(paths=saved_paths)
