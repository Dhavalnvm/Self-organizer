"""Product detection stage.

Thin wrapper around the SKU110K-trained YOLOv8 detector shipped in the repo
(`models/best.pt`). The model is class-agnostic — it localises individual
product facings; naming them is the classifier's job (see classifier.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Box:
    #A single detected product facing.

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def crop(self, image: np.ndarray) -> np.ndarray:
        #Return the image region for this box (clamped to image bounds).
        h, w = image.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return image[y1:y2, x1:x2]


class ProductDetector:
    #Detect product facings on a shelf image.

    def __init__(
        self,
        model_path: str | Path = config.MODEL_PATH,
        conf: float = config.DETECT_CONF,
        iou: float = config.DETECT_IOU,
        max_det: int = config.DETECT_MAX_DET,
        device: str = config.DEVICE,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device

    def detect(self, image: np.ndarray) -> list[Box]:
        #Run detection on a BGR/RGB numpy image and return boxes.
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        boxes: list[Box] = []
        for res in results:
            if res.boxes is None:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), float(c)))
        return boxes
