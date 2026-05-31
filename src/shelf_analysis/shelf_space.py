"""Shelf-space estimation stage.

Three light-weight geometric analyses on the detection boxes (no extra model):

* ``cluster_rows`` — group facings into shelf rows by 1-D clustering on the
  vertical centre of each box. Returns a row index per box and the row count.
* ``share_of_shelf`` — per-brand share of linear shelf space, approximated by the
  sum of box widths (≈ facing frontage) over total. This is the standard
  "Share of Shelf" (SOS) retail metric, here as a bbox proxy for true area.
* ``find_empty_slots`` — out-of-stock detection: per row, flag horizontal gaps
  between consecutive products that exceed one facing width, and estimate how
  many facings could fit in each gap. Used for On-Shelf Availability (OSA).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .detector import Box


@dataclass
class EmptySlot:
    """A run of missing product facings on a single shelf row."""

    row: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    est_missing_facings: int

    def to_dict(self) -> dict:
        return asdict(self)


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


def find_empty_slots(
    boxes: list[Box],
    row_ids: list[int],
    image_width: int,
    gap_ratio: float = 1.0,
    edge_ratio: float = 1.5,
) -> list[EmptySlot]:
    """Detect out-of-stock gaps within each shelf row.

    For each row, sort products by x and find horizontal gaps between consecutive
    facings that exceed ``gap_ratio * median_facing_width`` in that row. Each gap
    is reported as one ``EmptySlot`` with an estimated count of missing facings.

    Edge gaps (between the first/last product and the image border) are reported
    only when they are very large (``edge_ratio``) so we don't flag normal
    shelf-end whitespace — the row probably just ends there.
    """
    if not boxes:
        return []

    by_row: dict[int, list[Box]] = {}
    for b, r in zip(boxes, row_ids):
        by_row.setdefault(r, []).append(b)

    empties: list[EmptySlot] = []
    for r, row_boxes in by_row.items():
        if len(row_boxes) < 2:
            continue
        row_boxes = sorted(row_boxes, key=lambda b: b.x1)
        widths = np.array([b.width for b in row_boxes], dtype=float)
        median_w = float(np.median(widths))
        if median_w <= 0:
            continue
        threshold = gap_ratio * median_w
        ys = [b.y1 for b in row_boxes]; y1 = int(min(ys))
        y2 = int(max(b.y2 for b in row_boxes))

        # Internal gaps between consecutive facings.
        for left, right in zip(row_boxes[:-1], row_boxes[1:]):
            gap = right.x1 - left.x2
            if gap > threshold:
                n = max(1, int(round(gap / median_w)))
                empties.append(EmptySlot(
                    row=r, x1=int(left.x2), y1=y1, x2=int(right.x1), y2=y2,
                    width=int(gap), est_missing_facings=n,
                ))

        # Edge gaps (only flag large ones — small whitespace is normal).
        edge_threshold = edge_ratio * median_w
        left_edge = row_boxes[0].x1
        if left_edge > edge_threshold:
            n = max(1, int(round(left_edge / median_w)))
            empties.append(EmptySlot(
                row=r, x1=0, y1=y1, x2=int(left_edge), y2=y2,
                width=int(left_edge), est_missing_facings=n,
            ))
        right_edge = image_width - row_boxes[-1].x2
        if right_edge > edge_threshold:
            n = max(1, int(round(right_edge / median_w)))
            empties.append(EmptySlot(
                row=r, x1=int(row_boxes[-1].x2), y1=y1, x2=image_width, y2=y2,
                width=int(right_edge), est_missing_facings=n,
            ))
    return empties


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
