"""Retail shelf intelligence pipeline.

Modular stages: detection (YOLOv8) -> brand classification (CLIP zero-shot,
KNN fallback) -> OCR (EasyOCR) -> shelf-space estimation -> business metrics.
"""

from .pipeline import ShelfPipeline
from .detector import ProductDetector, Box
from .classifier import CLIPClassifier, KNNClassifier, build_classifier

__all__ = [
    "ShelfPipeline",
    "ProductDetector",
    "Box",
    "CLIPClassifier",
    "KNNClassifier",
    "build_classifier",
]
