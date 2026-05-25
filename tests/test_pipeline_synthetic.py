from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from recongrafica.pipeline import run_extract


def test_pipeline_with_synthetic_steam_like_chart_and_manual_anchors(tmp_path: Path):
    image_path = tmp_path / "chart.png"
    anchors_path = tmp_path / "anchors.json"
    truth_path = tmp_path / "truth.csv"
    out_prefix = tmp_path / "out" / "chart"

    image = _synthetic_chart()
    cv2.imwrite(str(image_path), image)
    anchors_path.write_text(
        json.dumps(
            {
                "y": [
                    {"text": "1,00€", "pixel": [65, 420]},
                    {"text": "2,00€", "pixel": [65, 120]},
                ],
                "x": [
                    {"text": "01/01/2025", "pixel": [120, 462]},
                    {"text": "11/01/2025", "pixel": [760, 462]},
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=11, freq="D").date.astype(str),
            "price": np.linspace(1.0, 2.0, 11),
        }
    ).to_csv(truth_path, index=False)

    result = run_extract(image_path, out_prefix, truth_csv=truth_path, anchors_path=anchors_path)

    assert Path(result["csv"]).exists()
    assert Path(result["json"]).exists()
    assert Path(result["debug"]).exists()
    assert Path(result["mask"]).exists()
    assert result["points"] >= 8
    assert result["metrics"]["series"]["overlap_points"] >= 8


def _synthetic_chart() -> np.ndarray:
    image = np.full((520, 900, 3), (42, 42, 42), dtype=np.uint8)
    plot_left, plot_top, plot_right, plot_bottom = 120, 100, 780, 430
    cv2.rectangle(image, (plot_left, plot_top), (plot_right, plot_bottom), (62, 62, 62), 1)
    for i in range(6):
        y = int(plot_top + i * (plot_bottom - plot_top) / 5)
        cv2.line(image, (plot_left, y), (plot_right, y), (56, 56, 56), 1)
    for i in range(6):
        x = int(plot_left + i * (plot_right - plot_left) / 5)
        cv2.line(image, (x, plot_top), (x, plot_bottom), (56, 56, 56), 1)
    points = []
    for x in range(plot_left, plot_right + 1, 8):
        t = (x - plot_left) / (plot_right - plot_left)
        y = plot_bottom - t * (plot_bottom - plot_top)
        points.append((x, int(y)))
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (70, 190, 90), 3)
    cv2.putText(image, "1,00E", (42, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(image, "2,00E", (42, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(image, "01/01/2025", (100, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(image, "11/01/2025", (700, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    return image
