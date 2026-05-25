from __future__ import annotations

from datetime import date, datetime

import numpy as np

from recongrafica.models import AxisAnchor, OCRResult, SeriesPoint, SignalPoint
from recongrafica.parsing import parse_date, parse_price


def anchors_from_ocr(results: list[OCRResult]) -> tuple[list[AxisAnchor], list[OCRResult]]:
    anchors: list[AxisAnchor] = []
    rejected: list[OCRResult] = []
    for result in results:
        cx, cy = result.box.center
        if result.axis == "y":
            price = parse_price(result.text)
            if price is None:
                rejected.append(result)
                continue
            anchors.append(AxisAnchor("y", cy, price, result.text, result.confidence))
        else:
            parsed_date = parse_date(result.text)
            if parsed_date is None:
                rejected.append(result)
                continue
            anchors.append(AxisAnchor("x", cx, parsed_date, result.text, result.confidence))
    return anchors, rejected


def reconstruct_series(signal: list[SignalPoint], anchors: list[AxisAnchor]) -> list[SeriesPoint]:
    x_anchors = [a for a in anchors if a.axis == "x" and isinstance(a.value, date)]
    y_anchors = [a for a in anchors if a.axis == "y" and isinstance(a.value, float)]
    if len(x_anchors) < 2:
        raise ValueError("Se necesitan al menos dos anclas de fecha en el eje X")
    if len(y_anchors) < 2:
        raise ValueError("Se necesitan al menos dos anclas de precio en el eje Y")
    if not signal:
        raise ValueError("No se detectaron puntos de senal visual")

    x_pixels = np.array([a.pixel for a in x_anchors], dtype=float)
    timestamps = np.array([_date_to_ordinal(a.value) for a in x_anchors], dtype=float)
    y_pixels = np.array([a.pixel for a in y_anchors], dtype=float)
    prices = np.array([float(a.value) for a in y_anchors], dtype=float)

    x_coef = np.polyfit(x_pixels, timestamps, deg=1)
    y_coef = np.polyfit(y_pixels, prices, deg=1)

    series: list[SeriesPoint] = []
    seen_dates: set[date] = set()
    for point in sorted(signal, key=lambda p: p.pixel_x):
        ordinal = int(round(float(np.polyval(x_coef, point.pixel_x))))
        parsed_date = date.fromordinal(ordinal)
        if parsed_date in seen_dates:
            continue
        seen_dates.add(parsed_date)
        price = float(np.polyval(y_coef, point.pixel_y))
        series.append(
            SeriesPoint(
                date=parsed_date,
                price=round(price, 6),
                pixel_x=point.pixel_x,
                pixel_y=round(float(point.pixel_y), 3),
                confidence=point.confidence,
            )
        )
    return series


def _date_to_ordinal(value: date) -> int:
    if isinstance(value, datetime):
        return value.date().toordinal()
    return value.toordinal()
