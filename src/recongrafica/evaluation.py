from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recongrafica.models import SeriesPoint
from recongrafica.parsing import cer, wer


def evaluate_series(series: list[SeriesPoint], truth_csv: str | Path | None) -> dict[str, Any]:
    if truth_csv is None:
        return {}
    truth_path = Path(truth_csv)
    if not truth_path.exists():
        raise FileNotFoundError(f"No existe el ground truth: {truth_path}")

    truth = pd.read_csv(truth_path)
    if not {"date", "price"}.issubset(truth.columns):
        raise ValueError("El ground truth debe contener columnas `date` y `price`")
    if not series:
        return {"series_points": 0, "truth_points": int(len(truth))}

    truth_dates = pd.to_datetime(truth["date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    truth_prices = truth["price"].astype(float).to_numpy()
    order = np.argsort(truth_dates)
    truth_dates = truth_dates[order]
    truth_prices = truth_prices[order]

    predicted_dates = np.array([pd.Timestamp(p.date).toordinal() for p in series], dtype=float)
    predicted_prices = np.array([p.price for p in series], dtype=float)
    valid = (predicted_dates >= truth_dates.min()) & (predicted_dates <= truth_dates.max())
    if not np.any(valid):
        return {
            "series_points": len(series),
            "truth_points": int(len(truth)),
            "overlap_points": 0,
        }

    expected_prices = np.interp(predicted_dates[valid], truth_dates, truth_prices)
    predicted_overlap = predicted_prices[valid]
    mse = mean_squared_error(expected_prices, predicted_overlap)
    return {
        "series_points": len(series),
        "truth_points": int(len(truth)),
        "overlap_points": int(valid.sum()),
        "mae": float(mean_absolute_error(expected_prices, predicted_overlap)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
    }


def evaluate_ocr_text(path: str | Path | None, predicted_texts: list[str] | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = data.get("ocr_text")
    if not payload:
        return {}
    expected = [str(item) for item in payload.get("expected", [])]
    predicted = [str(item) for item in payload.get("predicted", predicted_texts or [])]
    pairs = list(zip(expected, predicted))
    if not pairs:
        return {}
    return {
        "cer": float(np.mean([cer(e, p) for e, p in pairs])),
        "wer": float(np.mean([wer(e, p) for e, p in pairs])),
        "pairs": len(pairs),
    }
