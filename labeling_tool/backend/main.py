from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
from . import models, database
from .routers import tasks_router, images_router, models_router, static_router, export_router, system_router, analysis_router

# 创建数据库表
models.Base.metadata.create_all(bind=database.engine)

# 获取项目根目录（基于当前文件位置）
# 当前文件: labeling_tool/backend/main.py
# 项目根目录: 向上两级目录
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # 从 backend/main.py 到项目根目录
_frontend_static_dir = _project_root / "labeling_tool" / "frontend" / "static"
_frontend_index_file = _project_root / "labeling_tool" / "frontend" / "index.html"
_frontend_manual_file = _project_root / "labeling_tool" / "frontend" / "manual.html"

# 如果相对路径找不到，尝试从当前工作目录查找
if not _frontend_static_dir.exists():
    # 尝试从当前工作目录查找
    _cwd_static = Path.cwd() / "labeling_tool" / "frontend" / "static"
    if _cwd_static.exists():
        _frontend_static_dir = _cwd_static
        _frontend_index_file = Path.cwd() / "labeling_tool" / "frontend" / "index.html"
        _frontend_manual_file = Path.cwd() / "labeling_tool" / "frontend" / "manual.html"

app = FastAPI(title="Balloon/D1 Labeling Tool API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(tasks_router.router, prefix="/api") # 增加 /api 前缀
app.include_router(images_router.router, prefix="/api")
app.include_router(models_router.router, prefix="/api")
app.include_router(static_router.router, prefix="/api") # 文件访问也放在 /api 下
app.include_router(export_router.router, prefix="/api")
app.include_router(system_router.router, prefix="/api")
app.include_router(analysis_router.router, prefix="/api")

# 挂载前端静态文件
# 使用绝对路径，确保在Docker容器中也能正确找到文件
@app.get("/")
def read_root():
    return FileResponse(str(_frontend_index_file))

# 兼容 /index.html 访问
@app.get("/index.html")
def read_index():
    return FileResponse(str(_frontend_index_file))

# 兼容 /manual.html 访问
@app.get("/manual.html")
def read_manual():
    return FileResponse(str(_frontend_manual_file))

# 挂载本地静态资源 (离线模式)
# 访问路径: /static/vue.global.prod.min.js
# 使用绝对路径，确保在Docker容器中也能正确找到文件
app.mount("/static", StaticFiles(directory=str(_frontend_static_dir)), name="static")
app.mount("/api/static", StaticFiles(directory=str(_frontend_static_dir)), name="api-static")

# 兼容前端引用的相对路径（如果有）
