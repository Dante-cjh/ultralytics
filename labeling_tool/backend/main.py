from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from . import models, database
from .routers import tasks_router, images_router, models_router, static_router, export_router, system_router

# 创建数据库表
models.Base.metadata.create_all(bind=database.engine)

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

# 挂载前端静态文件
# 1. 挂载 static 目录（如果有 css/js 文件夹）
# app.mount("/static", StaticFiles(directory="labeling_tool/frontend/static"), name="static")

# 2. 根路径返回 index.html
@app.get("/")
def read_root():
    return FileResponse("labeling_tool/frontend/index.html")

# 兼容 /index.html 访问
@app.get("/index.html")
def read_index():
    return FileResponse("labeling_tool/frontend/index.html")

# 3. 挂载本地静态资源 (离线模式)
# 访问路径: /static/vue.global.prod.min.js
app.mount("/static", StaticFiles(directory="labeling_tool/frontend/static"), name="static")

# 兼容前端引用的相对路径（如果有）
