"""End-to-end orchestration: detect -> classify -> OCR -> shelf-space -> metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import config, io_utils, labels, shelf_space
from .detector import ProductDetector
from .visualize import annotate


class ShelfPipeline:
    """Run all stages and assemble the business-metrics dict + annotated image.

    The OCR stage is optional (it loads EasyOCR weights) so the pipeline can run
    detection + classification standalone in smoke tests.
    """

    def __init__(self, classifier, detector: ProductDetector | None = None, ocr=None,
                 ocr_brand_correct: bool = config.OCR_BRAND_CORRECTION) -> None:
        self.detector = detector or ProductDetector()
        self.classifier = classifier
        self.ocr = ocr   # ShelfOCR instance or None
        self.ocr_brand_correct = ocr_brand_correct

    def run(self, image_path: str | Path) -> tuple[dict, np.ndarray]:
        image_path = Path(image_path)
        image = io_utils.load_image(image_path)
        h = image.shape[0]

        # 1. Detect product facings.
        boxes = self.detector.detect(image)

        # 2. Classify each facing into a brand (CLIP / KNN).
        crops = [b.crop(image) for b in boxes]
        predictions = self.classifier.classify(crops)   # [(brand, score), ...]

        # 3. OCR shelf labels / price tags (optional).
        ocr_items = self.ocr.read(image) if self.ocr else []

        # 3a. OCR-assisted brand resolution (experimental, off by default): override
        #     the CLIP label when readable on-pack / tag text confidently names a brand.
        if ocr_items and self.ocr_brand_correct:
            ocr_votes = labels.ocr_brand_for_boxes(boxes, ocr_items)
            predictions = [
                (vote, 1.0) if vote is not None else pred
                for pred, vote in zip(predictions, ocr_votes)
            ]
        brand_labels = [b for b, _ in predictions]

        # 3b. Prices: clean price tags, and price-per-brand from column geometry.
        prices = labels.extract_prices(ocr_items) if ocr_items else []
        price_list = sorted({p["text"] for p in prices}, key=lambda t: int(t.lstrip(config.CURRENCY)))
        price_by_brand = labels.prices_by_brand(prices, boxes, brand_labels)
        # For a clean annotated image, overlay only price tags (with the ₹ symbol),
        # not the full noisy OCR dump.
        price_items = [
            replace(it, text=f"{config.CURRENCY}{config.PRICE_REGEX.match(it.text).group(1)}")
            for it in ocr_items if config.PRICE_REGEX.match(it.text)
        ]

        # 4. Shelf-space metrics.
        _, n_rows = shelf_space.cluster_rows(boxes, h)
        sos = shelf_space.share_of_shelf(boxes, brand_labels)

        # 5. Assemble metrics dict (superset of the assignment's required schema).
        brand_counts = dict(Counter(brand_labels).most_common())
        metrics = {
            "image_name": image_path.name,
            "total_products": len(boxes),
            "brands": brand_counts,
            "share_of_shelf": sos,
            "num_shelf_rows": n_rows,
            "ocr_labels": price_list,
            "price_by_brand": price_by_brand,
        }

        annotated = annotate(image, boxes, predictions, price_items)
        return metrics, annotated
