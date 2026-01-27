import cv2
import numpy as np
from pathlib import Path

def rect_correct(rect):
    """
    矫正矩形顶点顺序: [tl, tr, br, bl]
    (top-left, top-right, bottom-right, bottom-left)
    """
    pts = rect
    rect_new = np.zeros((4, 2), dtype="float32")
    
    # 按照 sum(x+y) 找左上(最小)和右下(最大)
    s = pts.sum(axis=1)
    rect_new[0] = pts[np.argmin(s)]      # Top-left
    rect_new[2] = pts[np.argmax(s)]      # Bottom-right
    
    # 按照 diff(y-x) 找右上(最小)和左下(最大)
    diff = np.diff(pts, axis=1)
    rect_new[1] = pts[np.argmin(diff)]   # Top-right
    rect_new[3] = pts[np.argmax(diff)]   # Bottom-left
    
    return rect_new

def wrap_perspective(img, rect):
    """进行透视变换"""
    (tl, tr, br, bl) = rect
    
    # 计算新图的宽度 (取两条宽边的最大值)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算新图的高度 (取两条高边的最大值)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def process_perspective_correction(image_path_str: str, points: list) -> str:
    """
    主处理函数
    Args:
        image_path_str: 图片路径
        points: 4个点的列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    Returns:
        新图片的路径
    """
    path = Path(image_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path_str}")
        
    # 读取图片 (使用 imdecode 支持中文路径)
    # np.fromfile 读取二进制数据
    img_data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path_str}")
        
    # 数据转换
    rect = np.array(points, dtype="float32")
    
    # 调试日志：打印关键信息
    print(f"DEBUG: Processing image {image_path_str}, size={img.shape}")
    print(f"DEBUG: Input points:\n{rect}")
    
    try:
        corrected_rect = rect_correct(rect)
        print(f"DEBUG: Corrected rect:\n{corrected_rect}")
        
        # 执行变换
        processed_image = wrap_perspective(img, corrected_rect)
        print(f"DEBUG: Output image size={processed_image.shape}")
        
        if processed_image.size == 0 or np.sum(processed_image) == 0:
             print("WARNING: Processed image is empty or all black!")
             
    except Exception as e:
        raise ValueError(f"Perspective transform failed: {e}")
    
    # 避免覆盖原图
    # 修改策略：保存到 runs/corrected_images 目录下，不污染原数据集
    # 获取项目根目录 (假设当前文件在 labeling_tool/backend/image_utils.py)
    project_root = Path(__file__).resolve().parent.parent.parent
    save_dir = project_root / "runs" / "corrected_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    new_filename = f"{path.stem}_corrected{path.suffix}"
    new_path = save_dir / new_filename
    
    # 保存 (使用 imencode + tofile 支持中文路径)
    is_success, buffer = cv2.imencode(path.suffix, processed_image)
    if is_success:
        buffer.tofile(str(new_path))
    else:
        raise ValueError("Failed to encode corrected image")
    
    return str(new_path)
