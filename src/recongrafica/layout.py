from __future__ import annotations

import cv2
import numpy as np

from recongrafica.models import Box, Layout


def detect_layout(image: np.ndarray) -> Layout:
    """Detecta layout Steam/CS2 con heuristica robusta y fallback por margenes."""
    height, width = image.shape[:2]
    fallback = _fallback_layout(width, height)

    axis_layout = _axis_grid_layout(image)
    if axis_layout is not None:
        return axis_layout

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 18), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 18)))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

    xs = np.where(vertical > 0)[1]
    ys = np.where(horizontal > 0)[0]
    if xs.size < 10 or ys.size < 10:
        return fallback

    left_candidates = xs[xs < width * 0.35]
    right_candidates = xs[xs > width * 0.55]
    top_candidates = ys[ys < height * 0.45]
    bottom_candidates = ys[ys > height * 0.45]
    if (
        left_candidates.size == 0
        or right_candidates.size == 0
        or top_candidates.size == 0
        or bottom_candidates.size == 0
    ):
        return fallback

    left = int(np.percentile(left_candidates, 80))
    right = int(np.percentile(right_candidates, 20))
    top = int(np.percentile(top_candidates, 80))
    bottom = int(np.percentile(bottom_candidates, 20))

    min_plot_width = width * 0.35
    min_plot_height = height * 0.25
    if right - left < min_plot_width or bottom - top < min_plot_height:
        return fallback

    plot = Box(left, top, right - left, bottom - top).clamp(width, height)
    return _layout_from_plot(plot, width, height)


def _axis_grid_layout(image: np.ndarray) -> Layout | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Ejes y ticks suelen ser grises: mas brillantes que el fondo y con baja saturacion.
    neutral = ((gray > 52) & (hsv[:, :, 1] < 80)).astype(np.uint8)
    vertical_counts = neutral.sum(axis=0)
    vertical_groups = _groups_from_indices(np.where(vertical_counts > height * 0.35)[0])
    if len(vertical_groups) < 2:
        return None

    centers = [(start + end) // 2 for start, end in vertical_groups]
    left_candidates = [x for x in centers if x < width * 0.4]
    right_candidates = [x for x in centers if x > width * 0.55]
    if not left_candidates or not right_candidates:
        return None

    left = max(left_candidates)
    right = min(right_candidates)
    if right - left < width * 0.35:
        return None

    axis_band = np.zeros(height, dtype=np.int32)
    for x in (left, right):
        x0 = max(0, x - 4)
        x1 = min(width, x + 5)
        axis_band += neutral[:, x0:x1].sum(axis=1).astype(np.int32)

    tick_rows = np.where(axis_band >= 3)[0]
    row_groups = _groups_from_indices(tick_rows)
    row_centers = [(start + end) // 2 for start, end in row_groups if end - start <= 4]
    if len(row_centers) < 2:
        return None

    top = min(row_centers)
    bottom = max(row_centers)
    if bottom - top < height * 0.35:
        return None

    plot = Box(left, top, right - left, bottom - top).clamp(width, height)
    return _layout_from_plot(plot, width, height)


def _groups_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        groups.append((start, prev))
        start = prev = current
    groups.append((start, prev))
    return groups


def _fallback_layout(width: int, height: int) -> Layout:
    plot = Box(
        x=int(width * 0.13),
        y=int(height * 0.11),
        width=int(width * 0.80),
        height=int(height * 0.68),
    ).clamp(width, height)
    return _layout_from_plot(plot, width, height)


def _layout_from_plot(plot: Box, width: int, height: int) -> Layout:
    x_axis_top = max(0, plot.bottom - int(height * 0.05))
    y_axis = Box(
        x=max(0, plot.x - int(width * 0.12)),
        y=plot.y,
        width=min(plot.x, int(width * 0.12)),
        height=plot.height,
    ).clamp(width, height)
    x_axis = Box(
        x=plot.x,
        y=x_axis_top,
        width=plot.width,
        height=max(1, min(height - x_axis_top, int(height * 0.23))),
    ).clamp(width, height)
    return Layout(plot_area=plot, x_axis_roi=x_axis, y_axis_roi=y_axis)
