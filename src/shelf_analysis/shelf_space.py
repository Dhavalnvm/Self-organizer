"""Shelf-space estimation stage.

Two light-weight geometric analyses on the detection boxes (no extra model):

* ``cluster_rows`` — group facings into shelf rows by 1-D clustering on the
  vertical centre of each box. Returns a row index per box and the row count.
* ``share_of_shelf`` — per-brand share of linear shelf space, approximated by the
  sum of box widths (≈ facing frontage) over total. This is the standard
  "Share of Shelf" (SOS) retail metric, here as a bbox proxy for true area.
"""

from __future__ import annotations

import numpy as np

from .detector import Box


def cluster_rows(boxes: list[Box], image_height: int) -> tuple[list[int], int]:
    """Assign each box a shelf-row id (top row = 0). Returns (row_ids, n_rows).

    Uses a simple gap-based split on sorted vertical centres: a new row starts
    when the vertical gap to the previous facing exceeds a fraction of the median
    facing height. This is robust to the number of rows without a fixed k.
    """
    if not boxes:
        return [], 0

    cys = np.array([b.cy for b in boxes], dtype=float)
    heights = np.array([b.height for b in boxes], dtype=float)
    order = np.argsort(cys)

    median_h = float(np.median(heights)) if len(heights) else image_height
    gap_threshold = 0.6 * median_h        # > 60% of a facing height => new row

    row_of = np.zeros(len(boxes), dtype=int)
    current_row = 0
    prev_cy = cys[order[0]]
    for rank, idx in enumerate(order):
        if rank > 0 and (cys[idx] - prev_cy) > gap_threshold:
            current_row += 1
        row_of[idx] = current_row
        prev_cy = cys[idx]
    return row_of.tolist(), current_row + 1


def share_of_shelf(boxes: list[Box], brands: list[str]) -> dict[str, str]:
    """Per-brand share of linear shelf frontage, as percentage strings."""
    if not boxes:
        return {}
    total_width = sum(b.width for b in boxes)
    if total_width <= 0:
        return {}
    width_by_brand: dict[str, float] = {}
    for box, brand in zip(boxes, brands):
        width_by_brand[brand] = width_by_brand.get(brand, 0.0) + box.width
    # Sort high -> low for readability.
    ordered = sorted(width_by_brand.items(), key=lambda kv: kv[1], reverse=True)
    return {brand: f"{(w / total_width) * 100:.1f}%" for brand, w in ordered}
