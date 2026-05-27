from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from recongrafica.image_io import ensure_parent
from recongrafica.models import AxisAnchor, Layout, OCRResult, SeriesPoint, SignalPoint


def export_csv(series: list[SeriesPoint], path: str | Path) -> None:
    ensure_parent(path)
    frame = pd.DataFrame([point.to_dict() for point in series])
    if frame.empty:
        frame = pd.DataFrame(columns=["date", "price", "pixel_x", "pixel_y", "confidence"])
    frame.to_csv(path, index=False)


def export_json(payload: dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_json_payload(
    image_shape: tuple[int, int, int],
    layout: Layout,
    ocr_results: list[OCRResult],
    anchors: list[AxisAnchor],
    rejected_ocr: list[OCRResult],
    signal: list[SignalPoint],
    series: list[SeriesPoint],
    metrics: dict[str, Any],
    baselines: dict[str, list[OCRResult]],
) -> dict[str, Any]:
    height, width = image_shape[:2]
    return {
        "image_metadata": {"width": width, "height": height, "channels": image_shape[2] if len(image_shape) > 2 else 1},
        "layout": layout.to_dict(),
        "axis_anchors": [anchor.to_dict() for anchor in anchors],
        "ocr_results": [result.to_dict() for result in ocr_results],
        "rejected_ocr": [result.to_dict() for result in rejected_ocr],
        "ocr_baselines": {name: [result.to_dict() for result in values] for name, values in baselines.items()},
        "signal_points": len(signal),
        "reconstructed_series": [point.to_dict() for point in series],
        "metrics": metrics,
    }


def save_debug_image(
    image,
    layout: Layout,
    ocr_results: list[OCRResult],
    anchors: list[AxisAnchor],
    signal: list[SignalPoint],
    path: str | Path,
) -> None:
    ensure_parent(path)
    debug = image.copy()
    _rectangle(debug, layout.plot_area, (60, 220, 60), 2)
    _rectangle(debug, layout.x_axis_roi, (220, 160, 60), 2)
    _rectangle(debug, layout.y_axis_roi, (60, 160, 220), 2)
    for result in ocr_results:
        if not _matches_anchor(result, anchors):
            continue
        color = (255, 180, 0) if result.axis == "x" else (0, 180, 255)
        _rectangle(debug, result.box, color, 1)
        cv2.putText(debug, result.text[:18], (result.box.x, max(12, result.box.y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for anchor in anchors:
        if anchor.axis == "x":
            cv2.circle(debug, (int(anchor.pixel), layout.x_axis_roi.y + 5), 5, (255, 0, 255), -1)
        else:
            cv2.circle(debug, (layout.y_axis_roi.right - 5, int(anchor.pixel)), 5, (255, 0, 255), -1)
    if signal:
        polyline = np.array(
            [[point.pixel_x, int(round(point.pixel_y))] for point in signal],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(debug, [polyline], False, (0, 255, 255), 1)
    cv2.imwrite(str(path), debug)


def save_mask(mask, path: str | Path) -> None:
    ensure_parent(path)
    cv2.imwrite(str(path), mask)


def _rectangle(image, box, color: tuple[int, int, int], thickness: int) -> None:
    cv2.rectangle(image, (box.x, box.y), (box.right, box.bottom), color, thickness)


def _matches_anchor(result: OCRResult, anchors: list[AxisAnchor]) -> bool:
    cx, cy = result.box.center
    pixel = cx if result.axis == "x" else cy
    return any(
        anchor.axis == result.axis
        and anchor.text == result.text
        and abs(anchor.pixel - pixel) <= 4
        for anchor in anchors
    )
