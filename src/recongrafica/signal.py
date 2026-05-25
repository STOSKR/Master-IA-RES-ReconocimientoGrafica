from __future__ import annotations

import cv2
import numpy as np

from recongrafica.image_io import crop
from recongrafica.models import Box, SignalPoint
from recongrafica.preprocessing import normalize_for_signal


def extract_price_signal(image: np.ndarray, plot_area: Box) -> tuple[list[SignalPoint], np.ndarray]:
    plot = crop(image, plot_area)
    smoothed = normalize_for_signal(plot)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    # Steam suele usar lineas azules/verdes con saturacion superior al fondo gris.
    saturated = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([140, 255, 255]))
    bright_saturated = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(saturated, bright_saturated)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _largest_components(mask, keep=3)

    points = _points_from_mask(mask, plot_area)
    if len(points) < max(5, plot_area.width // 20):
        edge_mask = _edge_fallback(plot)
        points = _points_from_mask(edge_mask, plot_area)
        mask = edge_mask
    return points, mask


def _largest_components(mask: np.ndarray, keep: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = [(idx, stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, count)]
    selected = {idx for idx, _ in sorted(areas, key=lambda item: item[1], reverse=True)[:keep]}
    output = np.zeros_like(mask)
    for idx in selected:
        output[labels == idx] = 255
    return output


def _points_from_mask(mask: np.ndarray, plot_area: Box) -> list[SignalPoint]:
    points: list[SignalPoint] = []
    height = mask.shape[0]
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        y = float(np.median(ys))
        confidence = min(1.0, ys.size / max(1, height * 0.04))
        points.append(SignalPoint(plot_area.x + x, plot_area.y + y, confidence))
    return points


def _edge_fallback(plot: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
