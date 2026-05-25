from datetime import date

from recongrafica.calibration import anchors_from_ocr, reconstruct_series
from recongrafica.models import Box, OCRResult, SignalPoint


def test_reconstruct_series_from_manual_like_ocr_results():
    ocr_results = [
        OCRResult("1,00€", 1.0, Box(7, 197, 6, 6), "y", "manual"),
        OCRResult("2,00€", 1.0, Box(7, 97, 6, 6), "y", "manual"),
        OCRResult("01/01/2025", 1.0, Box(97, 257, 6, 6), "x", "manual"),
        OCRResult("11/01/2025", 1.0, Box(197, 257, 6, 6), "x", "manual"),
    ]
    anchors, rejected = anchors_from_ocr(ocr_results)
    assert rejected == []

    series = reconstruct_series([SignalPoint(150, 150)], anchors)

    assert series[0].date == date(2025, 1, 6)
    assert round(series[0].price, 2) == 1.5
