from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./labeling_tool/data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class InferenceTask(Base):
    """
    一次推理任务（批次）
    """
    __tablename__ = "inference_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, index=True) # 任务名称，如 "20260112_Batch1"
    created_at = Column(DateTime, default=datetime.now)
    model_name = Column(String) # 使用的模型名称
    status = Column(String, default="completed") # processing, completed, failed
    folder_path = Column(String) # 原始图片文件夹路径
    
    # 进度统计
    total_images = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    
    # 类别信息
    classes = Column(JSON, default=[{"id": 0, "name": "object"}]) # 存储任务相关的类别列表
    
    # 关联
    images = relationship("ImageRecord", back_populates="task", cascade="all, delete-orphan")

class ImageRecord(Base):
    """
    单张图片记录
    """
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("inference_tasks.id"))
    file_path = Column(String, index=True) # 图片绝对路径
    file_name = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    status = Column(String, default="inferred") # inferred(已推理), corrected(已修订), exported(已导出)
    
    # 关联
    task = relationship("InferenceTask", back_populates="images")
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")

class Detection(Base):
    """
    检测框信息
    """
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("image_records.id"))
    
    class_id = Column(Integer)
    class_name = Column(String)
    confidence = Column(Float) # 如果是人工标注，置信度可以是 1.0
    
    # YOLO 格式坐标 (normalized center_x, center_y, w, h)
    x_center = Column(Float)
    y_center = Column(Float)
    width = Column(Float)
    height = Column(Float)
    
    is_manual = Column(Boolean, default=False) # 是否为人工添加/修改
    
    image = relationship("ImageRecord", back_populates="detections")

# 创建数据库表
def init_db():
    Base.metadata.create_all(bind=engine)
