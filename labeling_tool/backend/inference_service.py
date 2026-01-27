import sys
import os
import cv2
from pathlib import Path
from typing import List, Dict, Any

# 将项目根目录添加到 python path，以便能导入 D1_inference_with_sahi_v3
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from D1_inference_with_sahi_v3 import BalloonSAHIInference
from ultralytics.utils import LOGGER

class SahiInferenceService:
    def __init__(self):
        self.inference_engine = None
        self.current_model_path = None
        self.device = "cuda:0"
        # 缓存当前的配置参数，用于判断是否需要重新初始化
        self.current_config = {}

    @property
    def current_model(self):
        """兼容旧代码访问"""
        return self.inference_engine.detection_model if self.inference_engine else None

    def load_model(self, model_path: str, confidence_threshold: float = 0.25, device: str = "cuda:0"):
        """
        加载或更换模型 (使用 BalloonSAHIInference)
        """
        # 检查是否需要重新加载
        if (self.current_model_path == model_path and 
            self.inference_engine is not None and 
            self.device == device and 
            abs(self.current_config.get('conf', -1) - confidence_threshold) < 1e-6):
            return

        LOGGER.info(f"Initializing BalloonSAHIInference from {model_path} with conf={confidence_threshold} on {device}...")
        try:
            self.inference_engine = BalloonSAHIInference(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device
            )
            
            self.current_model_path = model_path
            self.device = device
            self.current_config = {'conf': confidence_threshold}
            LOGGER.info("BalloonSAHIInference initialized successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize inference engine: {e}")
            raise ValueError(f"Could not load model: {e}")

    def infer_slice(
        self, 
        image_path: str, 
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.15,
        overlap_width_ratio: float = 0.15,
        postprocess_type: str = "NMS",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5,
        min_box_area: int = 0,
        max_detections: int = 1000,
        conf_thres: float = 0.25,
        cross_class_nms_enabled: bool = True,
        cross_class_nms_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        执行 SAHI 切片推理 (直接调用 D1_inference_with_sahi_v3)
        """
        if not self.inference_engine:
            raise RuntimeError("Inference engine not initialized. Call load_model first.")

        # 调用 D1_inference_with_sahi_v3 中的 predict_image 方法
        # 注意：该方法返回的是一个字典，包含 'object_prediction_list' 等
        # 我们这里不需要它保存文件或可视化，只获取结果
        
        # 临时确保置信度阈值一致 (如果运行时传入的 conf_thres 与初始化不同)
        if hasattr(self.inference_engine, 'detection_model'):
            self.inference_engine.detection_model.confidence_threshold = conf_thres
            
        result_dict = self.inference_engine.predict_image(
            image_path=image_path,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=postprocess_type,
            postprocess_threshold=postprocess_match_threshold,
            postprocess_metric=postprocess_match_metric,
            cross_class_nms_enabled=cross_class_nms_enabled,
            cross_class_nms_threshold=cross_class_nms_threshold,
            min_box_area=min_box_area,
            max_detections=max_detections,
            visualize=False,  # 后端不需要生成可视化图
            save_txt=False,   # 后端不需要生成txt
            save_conf=False,
            save_dir=None
        )

        # 格式转换: 将 BalloonSAHIInference 的结果转换为后端需要的字典格式
        detections = []
        
        # 这里的 result_dict["object_prediction_list"] 已经被 D1_inference_with_sahi_v3 
        # 执行过所有的过滤逻辑（面积、跨类别NMS等），所以直接使用即可
        
        # 需要读取图片尺寸来归一化坐标
        img = cv2.imread(image_path)
        if img is None:
             # 如果 OpenCV 读取失败，尝试用 PIL 或其他方式，或者从 result 里的 shape 推断
             # 这里 D1 脚本里已经读过了，但没直接返回 shape。
             # 不过 predict_image 内部用的 read_image，我们这里再读一次获取宽高
             pass
        img_h, img_w = img.shape[:2]

        for pred in result_dict["detections"]:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = bbox
            
            # 转换为归一化中心坐标 (YOLO format)
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2

            detections.append({
                "class_id": pred.category.id,
                "class_name": pred.category.name,
                "confidence": float(pred.score.value),
                "x_center": float(x_center / img_w),
                "y_center": float(y_center / img_h),
                "width": float(width / img_w),
                "height": float(height / img_h)
            })
            
        return detections

# 单例模式
inference_service = SahiInferenceService()
