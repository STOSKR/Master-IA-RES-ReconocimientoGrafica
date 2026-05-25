from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return image


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def crop(image: np.ndarray, box: "Box") -> np.ndarray:
    from recongrafica.models import Box

    if not isinstance(box, Box):
        raise TypeError("box debe ser una instancia de Box")
    return image[box.y : box.bottom, box.x : box.right].copy()
