from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def clamp(self, image_width: int, image_height: int) -> "Box":
        x = max(0, min(self.x, image_width - 1))
        y = max(0, min(self.y, image_height - 1))
        right = max(x + 1, min(self.right, image_width))
        bottom = max(y + 1, min(self.bottom, image_height))
        return Box(x, y, right - x, bottom - y)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.right, self.bottom)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Layout:
    plot_area: Box
    x_axis_roi: Box
    y_axis_roi: Box

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "plot_area": self.plot_area.to_dict(),
            "x_axis_roi": self.x_axis_roi.to_dict(),
            "y_axis_roi": self.y_axis_roi.to_dict(),
        }


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    box: Box
    axis: Literal["x", "y"]
    source: Literal["paddle_preprocessed", "paddle_raw", "manual"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "axis": self.axis,
            "source": self.source,
        }


@dataclass(frozen=True)
class AxisAnchor:
    axis: Literal["x", "y"]
    pixel: float
    value: float | date
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        value: Any = self.value.isoformat() if isinstance(self.value, date) else self.value
        return {
            "axis": self.axis,
            "pixel": self.pixel,
            "value": value,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SignalPoint:
    pixel_x: int
    pixel_y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class SeriesPoint:
    date: date
    price: float
    pixel_x: int
    pixel_y: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "price": self.price,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "confidence": self.confidence,
        }
