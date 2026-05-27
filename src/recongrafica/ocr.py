from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from recongrafica.image_io import crop
from recongrafica.models import Box, Layout, OCRResult
from recongrafica.preprocessing import normalize_for_ocr


def read_manual_anchors(path: str | Path) -> list[OCRResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results: list[OCRResult] = []
    for axis in ("x", "y"):
        for item in data.get(axis, []):
            px, py = item["pixel"]
            box = Box(int(px) - 3, int(py) - 3, 6, 6)
            results.append(
                OCRResult(
                    text=str(item["text"]),
                    confidence=float(item.get("confidence", 1.0)),
                    box=box,
                    axis=axis,  # type: ignore[arg-type]
                    source="manual",
                )
            )
    return results


def run_ocr_baselines(image: np.ndarray, layout: Layout, languages: list[str] | None = None) -> dict[str, list[OCRResult]]:
    return {
        "preprocessed": run_paddle_ocr(image, layout, preprocessed=True, languages=languages),
        "raw": run_paddle_ocr(image, layout, preprocessed=False, languages=languages),
    }


def merge_ocr_results(baselines: dict[str, list[OCRResult]]) -> list[OCRResult]:
    merged: list[OCRResult] = []
    seen: set[tuple[str, str, int, int]] = set()
    for source_results in baselines.values():
        for result in source_results:
            cx, cy = result.box.center
            key = (result.axis, result.text.strip(), int(round(cx / 8)), int(round(cy / 8)))
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
    return merged


def run_paddle_ocr(
    image: np.ndarray,
    layout: Layout,
    preprocessed: bool,
    languages: list[str] | None = None,
) -> list[OCRResult]:
    PaddleOCR = _load_paddleocr()
    lang = (languages or ["en"])[0]
    engine = _create_paddle_engine(PaddleOCR, lang)
    source: Literal["paddle_preprocessed", "paddle_raw"] = "paddle_preprocessed" if preprocessed else "paddle_raw"

    results: list[OCRResult] = []
    for axis, roi in (("x", layout.x_axis_roi), ("y", layout.y_axis_roi)):
        roi_image = crop(image, roi)
        inference_image = normalize_for_ocr(roi_image) if preprocessed else roi_image
        scale_x = inference_image.shape[1] / roi.width
        scale_y = inference_image.shape[0] / roi.height
        raw = _run_paddle_engine(engine, inference_image)
        for line in _iter_paddle_lines(raw):
            points, (text, confidence) = line
            box = _box_from_points(points, roi, scale_x, scale_y)
            results.append(
                OCRResult(
                    text=str(text),
                    confidence=float(confidence),
                    box=box,
                    axis=axis,  # type: ignore[arg-type]
                    source=source,
                )
            )
    return results


def _load_paddleocr():
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR no esta instalado. Instala con `pip install -e \".[ocr]\"` "
            "o usa `--anchors data/anchors/captura.json` para anclas manuales."
        ) from exc
    return PaddleOCR


def _create_paddle_engine(PaddleOCR, lang: str):
    signature = inspect.signature(PaddleOCR)
    if "use_doc_unwarping" in signature.parameters:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    try:
        return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    except Exception as exc:
        if "show_log" not in str(exc):
            raise
        return PaddleOCR(use_angle_cls=True, lang=lang)


def _run_paddle_engine(engine, image: np.ndarray):
    try:
        if hasattr(engine, "predict"):
            return engine.predict(image)
        return engine.ocr(image, cls=True)
    except TypeError as exc:
        if "cls" not in str(exc):
            raise
        return engine.ocr(image)
    except NotImplementedError as exc:
        raise RuntimeError(
            "PaddleOCR esta instalado, pero PaddlePaddle ha fallado al ejecutar el modelo. "
            "En Windows esto suele ocurrir con versiones recientes de paddlepaddle 3.x. "
            "Usa Python 3.11 y reinstala con `pip install -e \".[dev,ocr]\"`, que fija PaddleOCR/PaddlePaddle 2.x."
        ) from exc


def _iter_paddle_lines(raw: list) -> list:
    lines = []
    for block in raw or []:
        if block is None:
            continue
        if hasattr(block, "to_dict"):
            block = block.to_dict()
        if isinstance(block, dict):
            texts = block.get("rec_texts", [])
            scores = block.get("rec_scores", [])
            polys = block.get("rec_polys", block.get("dt_polys", []))
            for points, text, confidence in zip(polys, texts, scores):
                lines.append((points, (text, confidence)))
            continue
        if block and isinstance(block[0], list) and len(block[0]) == 2 and isinstance(block[0][1], tuple):
            lines.extend(block)
        elif len(block) == 2 and isinstance(block[1], tuple):
            lines.append(block)
    return lines


def _box_from_points(points: list[list[float]], roi: Box, scale_x: float, scale_y: float) -> Box:
    xs = [p[0] / scale_x for p in points]
    ys = [p[1] / scale_y for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Box(
        x=int(round(roi.x + x0)),
        y=int(round(roi.y + y0)),
        width=max(1, int(round(x1 - x0))),
        height=max(1, int(round(y1 - y0))),
    )
