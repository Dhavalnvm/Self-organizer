"""OCR stage — read shelf labels / price tags with EasyOCR.

Price tags sit on the shelf-edge strips rather than on the products, so we OCR
the whole image and then keep text that looks like a price/weight/label. We
return both the raw detections (for visualization) and a price-filtered list of
strings (for the business-metrics output).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class OCRItem:
    text: str
    conf: float
    box: list[tuple[int, int]]   # 4 polygon points (EasyOCR order)


class ShelfOCR:
    def __init__(
        self,
        langs: list[str] = config.OCR_LANGS,
        min_conf: float = config.OCR_MIN_CONF,
        use_gpu: bool = config.DEVICE == "cuda",
    ) -> None:
        import easyocr

        self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        self.min_conf = min_conf

    def read(self, image: np.ndarray) -> list[OCRItem]:
        # All text detections above the confidence threshold.
        # Price/brand parsing of these items lives in ``labels.py``.
        items: list[OCRItem] = []
        for box, text, conf in self.reader.readtext(image):
            if conf < self.min_conf:
                continue
            text = text.strip()
            if text:
                pts = [(int(x), int(y)) for x, y in box]
                items.append(OCRItem(text=text, conf=float(conf), box=pts))
        return items
