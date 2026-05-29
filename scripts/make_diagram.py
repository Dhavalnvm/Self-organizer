"""Render a static PNG block diagram of the pipeline (no extra dependencies
beyond matplotlib). Saves to docs/architecture.png.

    python scripts/make_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"


def box(ax, xy, w, h, text, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor="#333", facecolor=color))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box(ax, (0.3, 4.6), 2.0, 0.9, "Shelf image", "#e8eef7")
    box(ax, (3.0, 4.6), 2.4, 0.9, "Detection\nYOLOv8 (SKU110K)", "#cfe3ff")
    box(ax, (6.1, 5.0), 3.4, 0.9, "Brand classifier\nCLIP zero-shot (primary)", "#d6f5d6")
    box(ax, (6.1, 3.9), 3.4, 0.8, "DINOv2/ResNet18 + KNN (fallback)", "#eafaea")
    box(ax, (3.0, 3.0), 2.4, 0.9, "OCR\nEasyOCR price/labels", "#ffe9cc")
    box(ax, (3.0, 1.5), 2.4, 0.9, "Shelf-space\nrows + Share-of-Shelf", "#f3e0ff")
    box(ax, (6.1, 1.9), 3.4, 0.9, "Aggregate metrics\nJSON + annotated image", "#ffd9d9")

    arrow(ax, 2.3, 5.05, 3.0, 5.05)
    arrow(ax, 5.4, 5.05, 6.1, 5.25)
    arrow(ax, 5.4, 4.9, 6.1, 4.3)
    arrow(ax, 1.3, 4.6, 4.2, 3.9)      # image -> OCR
    arrow(ax, 4.2, 4.6, 4.2, 3.9)      # detection -> OCR region
    arrow(ax, 4.2, 4.6, 4.2, 2.4)      # detection -> shelf-space
    arrow(ax, 7.8, 5.0, 7.8, 2.8)      # classifier -> aggregate
    arrow(ax, 5.4, 3.45, 6.1, 2.6)     # ocr -> aggregate
    arrow(ax, 5.4, 1.95, 6.1, 2.3)     # shelf-space -> aggregate

    ax.set_title("Retail Shelf Intelligence Pipeline", fontsize=13, weight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
