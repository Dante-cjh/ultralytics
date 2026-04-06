"""
弹片 / 着弹点统计分析。
笛卡尔坐标系：原点左下，x 向右，y 向上。

轴向规则：
  飞散角（dispersion）：立式看 Y（高低），卧式看 X（水平）。
  方向角（azimuth）  ：立式看 X（水平偏差），卧式看 Y（高低偏差）。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

CenterPx = Tuple[float, float]


def calc_dispersion_angle(
    centers: Sequence[CenterPx],
    pixel_width: float,
    pixel_height: float,
    real_width: float,
    real_height: float,
    distance: float,
    is_vertical: bool,
) -> float:
    """飞散角（度）。全角：2 * arctan((范围/2) / 距离)。
    仅在样本数 >= 10 时去除首尾各 5% 的离群点，避免小样本时过度过滤导致结果为 0。
    """
    if not centers or distance <= 0:
        return 0.0
    if pixel_width <= 0 or pixel_height <= 0:
        return 0.0

    if is_vertical:
        # 立式：飞散方向为高低（Y 方向）
        y_coords = sorted(float(cy) for _, cy in centers)
        num_points = len(y_coords)
        if num_points >= 10:
            exclude_count = max(1, int(num_points * 0.05))
            filtered_y = y_coords[exclude_count:-exclude_count] if num_points > 2 * exclude_count else y_coords
        else:
            filtered_y = y_coords
        if not filtered_y:
            return 0.0
        y_range_pixels = max(filtered_y) - min(filtered_y)
        span_meters = (y_range_pixels / pixel_height) * real_height
    else:
        # 卧式：飞散方向为水平（X 方向）
        x_coords = sorted(float(cx) for cx, _ in centers)
        num_points = len(x_coords)
        if num_points >= 10:
            exclude_count = max(1, int(num_points * 0.05))
            filtered_x = x_coords[exclude_count:-exclude_count] if num_points > 2 * exclude_count else x_coords
        else:
            filtered_x = x_coords
        if not filtered_x:
            return 0.0
        x_range_pixels = max(filtered_x) - min(filtered_x)
        span_meters = (x_range_pixels / pixel_width) * real_width

    dispersion_angle = math.atan((span_meters / 2) / distance) * 2 * 180 / math.pi
    return round(dispersion_angle, 2)


def calc_azimuth_angle(
    centers: Sequence[CenterPx],
    pixel_width: float,
    pixel_height: float,
    real_width: float,
    real_height: float,
    distance: float,
    is_vertical: bool,
    baseline_position: float = 0.0,
) -> float:
    """方向角（度），弹群均值中心相对基准线（或图像中心）的偏角。

    轴向规则（与前端散点图基准线一致）：
      - 立式（is_vertical=True）：方向角 = 水平方向（X）偏差；
        baseline_position 为 X 像素坐标，≤0 时取图像水平中心。
      - 卧式（is_vertical=False）：方向角 = 垂直方向（Y）偏差；
        baseline_position 为 Y 像素坐标（笛卡尔，原点左下），≤0 时取图像垂直中心。

    均值（而非中位数）作为弹群中心，符合弹道统计惯例。
    """
    if len(centers) < 2 or distance <= 0:
        return 0.0
    if pixel_width <= 0 or pixel_height <= 0:
        return 0.0

    if is_vertical:
        # 立式：方向角衡量水平（X）偏差
        x_vals = [float(cx) for cx, _ in centers]
        mean_x = sum(x_vals) / len(x_vals)

        reference_x = baseline_position if baseline_position > 0 else pixel_width / 2

        distance_pixels = abs(mean_x - reference_x)
        distance_meters = (distance_pixels / pixel_width) * real_width
        azimuth_angle = math.atan(distance_meters / distance) * 180 / math.pi
        if mean_x < reference_x:
            azimuth_angle = -azimuth_angle
    else:
        # 卧式：方向角衡量垂直（Y）偏差
        y_vals = [float(cy) for _, cy in centers]
        mean_y = sum(y_vals) / len(y_vals)

        reference_y = baseline_position if baseline_position > 0 else pixel_height / 2

        distance_pixels = abs(mean_y - reference_y)
        distance_meters = (distance_pixels / pixel_height) * real_height
        azimuth_angle = math.atan(distance_meters / distance) * 180 / math.pi
        if mean_y < reference_y:
            azimuth_angle = -azimuth_angle

    return round(azimuth_angle, 2)


def calc_fragment_density(
    centers: Sequence[CenterPx],
    real_width: float,
    real_height: float,
) -> float:
    """破片密度：个 / m²。"""
    if not centers:
        return 0.0
    area_m2 = real_width * real_height
    if area_m2 <= 0:
        return 0.0
    return round(len(centers) / area_m2, 2)


def run_ballistic_analysis(
    centers_px: List[CenterPx],
    pixel_width: float,
    pixel_height: float,
    real_width_m: float,
    real_height_m: float,
    distance_m: float,
    is_vertical: bool,
    baseline_px: float = 0.0,
    hole_areas_px: Optional[List[float]] = None,
) -> dict:
    """汇总计算，供 API 使用。"""
    centers = [(float(cx), float(cy)) for cx, cy in centers_px]
    dispersion = calc_dispersion_angle(
        centers, pixel_width, pixel_height, real_width_m, real_height_m, distance_m, is_vertical
    )
    azimuth = calc_azimuth_angle(
        centers,
        pixel_width,
        pixel_height,
        real_width_m,
        real_height_m,
        distance_m,
        is_vertical,
        baseline_px,
    )
    density = calc_fragment_density(centers, real_width_m, real_height_m)

    out = {
        "count": len(centers),
        "dispersion_angle_deg": dispersion,
        "azimuth_angle_deg": azimuth,
        "fragment_density_per_m2": density,
        "max_hole_area_px": None,
        "min_hole_area_px": None,
    }
    if hole_areas_px and len(hole_areas_px) > 0:
        areas = [float(a) for a in hole_areas_px]
        out["max_hole_area_px"] = round(max(areas), 2)
        out["min_hole_area_px"] = round(min(areas), 2)
    return out
