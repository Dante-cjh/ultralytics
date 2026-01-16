import subprocess
import sys
import os
import torch
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from pathlib import Path

router = APIRouter(
    prefix="/system",
    tags=["system"]
)

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

class PathResponse(BaseModel):
    path: str
    
class PathsResponse(BaseModel):
    paths: List[str]

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
