from __future__ import annotations

import cv2
import numpy as np

from recongrafica.image_io import crop
from recongrafica.models import Box, SignalPoint
from recongrafica.preprocessing import normalize_for_signal


def extract_price_signal(
    image: np.ndarray,
    plot_area: Box,
    y_quantile: float = 0.5,
) -> tuple[list[SignalPoint], np.ndarray]:
    if not 0 <= y_quantile <= 1:
        raise ValueError("y_quantile debe estar entre 0 y 1")

    plot = crop(image, plot_area)
    smoothed = normalize_for_signal(plot)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    score = _price_line_score(smoothed, hsv)

    # En capturas Steam el fondo puede ser azul/gris y tener saturacion moderada.
    # Primero aislamos la linea de precio verde/amarilla para no capturar el fondo
    # ni las barras azules de volumen.
    mask = cv2.inRange(hsv, np.array([35, 70, 55]), np.array([90, 255, 255]))
    mask = cv2.bitwise_and(mask, (score > 20).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = _largest_components(mask, keep=2)

    points = _points_from_weighted_mask(mask, score, plot_area, y_quantile)
    if len(points) < max(5, plot_area.width // 20):
        fallback_mask = _saturated_fallback(hsv)
        fallback_mask = cv2.bitwise_and(fallback_mask, (score > 12).astype(np.uint8) * 255)
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel)
        fallback_mask = _largest_components(fallback_mask, keep=3)
        fallback_points = _points_from_weighted_mask(fallback_mask, score, plot_area, y_quantile)
        if len(fallback_points) >= len(points):
            points = fallback_points
            mask = fallback_mask
    if len(points) < max(5, plot_area.width // 20):
        edge_mask = _edge_fallback(plot)
        edge_points = _points_from_weighted_mask(edge_mask, score, plot_area, y_quantile)
        if len(edge_points) >= len(points):
            points = edge_points
            mask = edge_mask
    return points, _centerline_mask(points, plot_area, mask.shape)


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


def _points_from_weighted_mask(
    mask: np.ndarray,
    score: np.ndarray,
    plot_area: Box,
    y_quantile: float,
) -> list[SignalPoint]:
    points: list[SignalPoint] = []
    height = mask.shape[0]
    previous_y: float | None = None
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        runs = _split_runs(ys)
        if previous_y is not None and len(runs) > 1:
            ys = min(runs, key=lambda run: abs(float(np.mean(run)) - previous_y))
        weights = score[ys, x].astype(float)
        if np.sum(weights) > 0:
            center_y = float(np.average(ys, weights=weights))
            quantile_y = float(np.quantile(ys, y_quantile))
            y = center_y if y_quantile == 0.5 else float((center_y + quantile_y) / 2)
        else:
            y = float(np.quantile(ys, y_quantile))
        previous_y = y
        confidence = min(1.0, ys.size / max(1, height * 0.04))
        points.append(SignalPoint(plot_area.x + x, plot_area.y + y, confidence))
    return points


def _edge_fallback(plot: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _saturated_fallback(hsv: np.ndarray) -> np.ndarray:
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    sat_threshold = max(90, int(np.percentile(saturation, 96)))
    val_threshold = max(55, int(np.percentile(value, 70)))
    return cv2.inRange(
        hsv,
        np.array([0, sat_threshold, val_threshold]),
        np.array([179, 255, 255]),
    )


def _price_line_score(image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    green_dominance = green - np.maximum(red, blue)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    score = green_dominance + (saturation // 5) + (value // 10)
    score[green_dominance < 8] = 0
    return np.clip(score, 0, 255).astype(np.uint8)


def _split_runs(indices: np.ndarray) -> list[np.ndarray]:
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return [run for run in np.split(indices, breaks) if run.size > 0]


def _centerline_mask(points: list[SignalPoint], plot_area: Box, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    if not points:
        return output
    polyline = np.array(
        [
            [point.pixel_x - plot_area.x, int(round(point.pixel_y - plot_area.y))]
            for point in points
        ],
        dtype=np.int32,
    )
    cv2.polylines(output, [polyline], isClosed=False, color=255, thickness=1)
    return output
