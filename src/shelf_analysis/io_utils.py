"""Small I/O helpers: load images, save JSON and annotated images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    #Load an image as a BGR numpy array (OpenCV native).
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def list_images(path: str | Path) -> list[Path]:
    #Return image paths for a file or directory input.
    p = Path(path)
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"Input path does not exist: {path}")


def save_json(data: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_image(image: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), image)
