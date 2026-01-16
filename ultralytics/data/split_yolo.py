# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
from glob import glob
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils import TQDM


def bbox_iof_yolo(bbox1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    计算YOLO格式边界框的IoF (Intersection over Foreground)
    
    Args:
        bbox1 (np.ndarray): 边界框坐标，形状为 (N, 4)，格式为 [x_min, y_min, x_max, y_max]
        bbox2 (np.ndarray): 窗口边界框坐标，形状为 (M, 4)，格式为 [x_min, y_min, x_max, y_max]
        eps (float, optional): 防止除零的小值
    
    Returns:
        (np.ndarray): IoF分数，形状为 (N, M)
    
    Notes:
        IoF = Intersection / Area(bbox1)
    """
    # 计算交集
    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])  # 左上角
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])   # 右下角
    wh = np.clip(rb - lt, 0, np.inf)
    intersection = wh[..., 0] * wh[..., 1]
    
    # 计算bbox1的面积
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area1 = area1[:, None]
    area1 = np.clip(area1, eps, np.inf)
    
    # 计算IoF
    iof = intersection / area1
    if iof.ndim == 1:
        iof = iof[..., None]
    return iof


def load_yolo_format(data_root: str, split: str = "train") -> list[dict[str, Any]]:
    """
    加载YOLO格式数据集的标注和图像信息
    
    Args:
        data_root (str): 数据根目录
        split (str, optional): 数据集分割，可以是 'train' 或 'val'
    
    Returns:
        (list[dict[str, Any]]): 包含图像信息的标注字典列表
    
    Notes:
        数据集目录结构:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if len(lb) > 0:
                lb = np.array(lb, dtype=np.float32)
            else:
                lb = np.zeros((0, 5), dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(
    im_size: tuple[int, int],
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
    im_rate_thr: float = 0.6,
    eps: float = 0.01,
) -> np.ndarray:
    """
    获取用于图像裁剪的滑动窗口坐标
    
    Args:
        im_size (tuple[int, int]): 原始图像尺寸, (H, W)
        crop_sizes (tuple[int, ...], optional): 裁剪窗口尺寸
        gaps (tuple[int, ...], optional): 裁剪间隔
        im_rate_thr (float, optional): 窗口面积与图像面积的阈值
        eps (float, optional): 数学运算的epsilon值
    
    Returns:
        (np.ndarray): 窗口坐标数组，形状为 (N, 4)，每行为 [x_start, y_start, x_stop, y_stop]
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]:
    """
    根据IoF阈值获取每个窗口中的目标
    
    Args:
        anno (dict[str, Any]): 标注字典
        windows (np.ndarray): 窗口坐标数组
        iof_thr (float, optional): IoF阈值
    
    Returns:
        (list[np.ndarray]): 每个窗口中的标签列表
    """
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        # YOLO格式: class_id x_center y_center width height (归一化)
        # 转换为绝对坐标: x_min, y_min, x_max, y_max
        boxes = label[:, 1:].copy()
        boxes[:, 0] *= w  # x_center
        boxes[:, 1] *= h  # y_center
        boxes[:, 2] *= w  # width
        boxes[:, 3] *= h  # height
        
        # 转换为 [x_min, y_min, x_max, y_max]
        x_min = boxes[:, 0] - boxes[:, 2] / 2
        y_min = boxes[:, 1] - boxes[:, 3] / 2
        x_max = boxes[:, 0] + boxes[:, 2] / 2
        y_max = boxes[:, 1] + boxes[:, 3] / 2
        
        bbox_xyxy = np.stack([x_min, y_min, x_max, y_max], axis=1)
        
        # 计算IoF
        iofs = bbox_iof_yolo(bbox_xyxy, windows)
        
        # 为每个窗口选择目标
        window_anns = []
        for i in range(len(windows)):
            mask = iofs[:, i] >= iof_thr
            if mask.any():
                window_anns.append(label[mask])
            else:
                window_anns.append(np.zeros((0, 5), dtype=np.float32))
        return window_anns
    else:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(len(windows))]


def crop_and_save(
    anno: dict[str, Any],
    windows: np.ndarray,
    window_objs: list[np.ndarray],
    im_dir: str,
    lb_dir: str,
    allow_background_images: bool = True,
) -> None:
    """
    裁剪图像并为每个窗口保存新标签
    
    Args:
        anno (dict[str, Any]): 标注字典，包含 'filepath', 'label', 'ori_size' 键
        windows (np.ndarray): 窗口坐标数组，形状为 (N, 4)
        window_objs (list[np.ndarray]): 每个窗口内的标签列表
        im_dir (str): 图像输出目录路径
        lb_dir (str): 标签输出目录路径
        allow_background_images (bool, optional): 是否包含没有标签的背景图像
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    h, w = anno["ori_size"]
    
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        label = window_objs[i]
        has_objects = len(label) > 0
        
        # 保存图像（有目标或允许背景图像）
        if has_objects or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
            
            # 关键修复：无论是否有目标，都创建标签文件
            label_file = Path(lb_dir) / f"{new_name}.txt"
            
            if has_objects:
                # 有目标：转换并保存标签
                new_label = label.copy()
                # YOLO格式: class_id x_center y_center width height (归一化)
                # 先转换为绝对坐标
                new_label[:, 1] *= w  # x_center
                new_label[:, 2] *= h  # y_center
                new_label[:, 3] *= w  # width
                new_label[:, 4] *= h  # height
                
                # 调整为窗口坐标
                new_label[:, 1] -= x_start  # x_center
                new_label[:, 2] -= y_start  # y_center
                
                # 归一化到新窗口
                new_label[:, 1] /= pw  # x_center
                new_label[:, 2] /= ph  # y_center
                new_label[:, 3] /= pw  # width
                new_label[:, 4] /= ph  # height
                
                # 裁剪坐标到有效范围 [0, 1]，避免边缘目标坐标超出范围
                new_label[:, 1:] = np.clip(new_label[:, 1:], 0, 1)
                
                # 过滤掉无效的标注（宽高接近0的）
                valid_mask = (new_label[:, 3] > 0.01) & (new_label[:, 4] > 0.01)
                new_label = new_label[valid_mask]
                
                # 保存标签
                with open(label_file, "w", encoding="utf-8") as f:
                    for lb in new_label:
                        formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                        f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
            else:
                # 没有目标：创建空标签文件（负样本）
                with open(label_file, "w", encoding="utf-8") as f:
                    pass  # 创建空文件


def split_images_and_labels(
    data_root: str,
    save_dir: str,
    split: str = "train",
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
) -> None:
    """
    为给定的数据集分割切分图像和标签
    
    Args:
        data_root (str): 数据集根目录
        save_dir (str): 保存切分数据集的目录
        split (str, optional): 数据集分割，可以是 'train' 或 'val'
        crop_sizes (tuple[int, ...], optional): 裁剪尺寸元组
        gaps (tuple[int, ...], optional): 裁剪间隔元组
    
    Notes:
        数据集目录结构:
            - data_root
                - images
                    - split
                - labels
                    - split
        输出目录结构:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_format(data_root, split=split)
    for anno in TQDM(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """
    使用多个缩放比例切分YOLO格式数据集的训练集和验证集
    
    Args:
        data_root (str): 数据集根目录
        save_dir (str): 保存切分数据集的目录
        crop_size (int, optional): 基础裁剪尺寸
        gap (int, optional): 基础裁剪间隔
        rates (tuple[float, ...], optional): crop_size和gap的缩放比例
    
    Notes:
        数据集目录结构:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        输出目录结构:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in {"train", "val"}:
        split_images_and_labels(data_root, save_dir, split, tuple(crop_sizes), tuple(gaps))


if __name__ == "__main__":
    # 示例用法
    split_trainval(
        data_root="/home/cjh/mmdetection/data/balloon/yolo_format",
        save_dir="/home/cjh/mmdetection/data/balloon/yolo_format_slice"
    )
