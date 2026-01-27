import cv2
import numpy as np
import os
from pathlib import Path

def create_grid_image(width=800, height=600, grid_size=50):
    # 创建白色背景
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 绘制网格
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), (220, 220, 220), 1)
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), (220, 220, 220), 1)
        
    # 绘制黑色粗边框（这就是你需要矫正的目标边缘）
    cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 0, 0), 10)
    
    # 绘制四个角的标记
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 左上 TL (红)
    cv2.circle(img, (50, 50), 30, (0, 0, 255), -1)
    cv2.putText(img, "TL", (20, 110), font, 1.5, (0, 0, 255), 3)
    
    # 右上 TR (绿)
    cv2.circle(img, (width-50, 50), 30, (0, 255, 0), -1)
    cv2.putText(img, "TR", (width-120, 110), font, 1.5, (0, 255, 0), 3)
    
    # 右下 BR (蓝)
    cv2.circle(img, (width-50, height-50), 30, (255, 0, 0), -1)
    cv2.putText(img, "BR", (width-120, height-20), font, 1.5, (255, 0, 0), 3)
    
    # 左下 BL (黄)
    cv2.circle(img, (50, height-50), 30, (0, 255, 255), -1)
    cv2.putText(img, "BL", (20, height-20), font, 1.5, (0, 255, 255), 3)
    
    # 中心文字
    text = "Correction Test"
    (tw, th), _ = cv2.getTextSize(text, font, 2, 4)
    cv2.putText(img, text, ((width-tw)//2, height//2), font, 2, (0, 0, 0), 4)
    
    return img

def warp_image(img, src_points, dst_points, output_size):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 使用灰色填充背景，模拟拍摄环境
    warped = cv2.warpPerspective(img, M, output_size, borderValue=(128, 128, 128))
    return warped

def main():
    # 输出目录: data/test/perspective_correction
    output_dir = Path("data/test/perspective_correction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    w, h = 800, 600
    base_img = create_grid_image(w, h)
    
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 1. 模拟梯形 (后仰视角)
    dst_pts_trap = np.float32([
        [200, 100],   # TL 向内向下
        [w-200, 100], # TR 向内向下
        [w-50, h-50], # BR 稍向内
        [50, h-50]    # BL 稍向内
    ])
    img_trap = warp_image(base_img, src_pts, dst_pts_trap, (w, h))
    cv2.imwrite(str(output_dir / "01_tilted_back.jpg"), img_trap)
    
    # 2. 模拟侧偏 (右侧远)
    dst_pts_side = np.float32([
        [50, 50],     # TL
        [w-150, 150], # TR 远
        [w-150, h-150],# BR 远
        [50, h-50]    # BL
    ])
    img_side = warp_image(base_img, src_pts, dst_pts_side, (w, h))
    cv2.imwrite(str(output_dir / "02_tilted_right.jpg"), img_side)
    
    # 3. 复杂任意角度
    dst_pts_complex = np.float32([
        [250, 150],   # TL
        [700, 50],    # TR
        [650, 500],   # BR
        [100, 450]    # BL
    ])
    img_complex = warp_image(base_img, src_pts, dst_pts_complex, (800, 600))
    cv2.imwrite(str(output_dir / "03_complex_angle.jpg"), img_complex)
    
    print(f"Successfully generated 3 test images in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
