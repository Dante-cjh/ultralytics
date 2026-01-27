from fastapi import APIRouter
from fastapi.responses import FileResponse
import os
from pathlib import Path

router = APIRouter(
    tags=["static"]
)

@router.get("/file/{file_path:path}")
def get_file(file_path: str):
    """
    提供本地文件访问（用于前端显示图片）
    注意：生产环境应使用 Nginx 代理，这里为了本地开发方便
    """
    # 简单的安全检查，防止访问系统关键文件
    # 在 Windows 下，路径可能包含盘符，需要小心处理
    path = Path(file_path)
    if not path.exists():
         # 尝试解码 url encoded path 如果有必要，FastAPI通常会自动处理
         return {"error": "File not found"}
         
    return FileResponse(file_path)
