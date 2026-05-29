"""Annotated-image rendering: product boxes + brand labels + OCR overlays."""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .detector import Box
from .ocr import OCRItem


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    # Draw text with a filled background chip for legibility.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - base - 2)
    cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + base + 2), color, -1)
    cv2.putText(img, text, (x + 2, y_top + th + 1), font, scale, config.TEXT_COLOR, thick, cv2.LINE_AA)


def annotate(
    image: np.ndarray,
    boxes: list[Box],
    brands: list[tuple[str, float]],
    ocr_items: list[OCRItem] | None = None,
) -> np.ndarray:
    #Return a copy of ``image`` with detection and OCR overlays drawn.
    out = image.copy()

    # Product boxes + brand labels.
    for box, (brand, score) in zip(boxes, brands):
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), config.BOX_COLOR, 2)
        _draw_label(out, f"{brand} {score:.2f}", x1, y1, config.BOX_COLOR)

    # OCR price/label overlays.
    for it in ocr_items or []:
        pts = np.array(it.box, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=config.OCR_COLOR, thickness=2)
        x, y = it.box[0]
        _draw_label(out, it.text, int(x), int(y), config.OCR_COLOR)

    return out
