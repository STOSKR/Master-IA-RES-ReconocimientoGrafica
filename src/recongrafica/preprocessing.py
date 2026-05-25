from __future__ import annotations

import cv2
import numpy as np


def normalize_for_ocr(image: np.ndarray, scale: int = 3) -> np.ndarray:
    """Preprocesado conservador para textos pequenos de ejes."""
    if scale < 1:
        raise ValueError("scale debe ser >= 1")

    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=7)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def normalize_for_signal(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), 0)
