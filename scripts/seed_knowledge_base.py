"""Crop detected products from shelf images and save them into the knowledge base. Builds a labelled gallery the KNN fallback can use.

    python scripts/seed_knowledge_base.py --input data/test_images
    python scripts/seed_knowledge_base.py --input data/test_images/img_1.jpg --min-score 0.27

Output layout mirrors the existing repo:

    data/knowledge_base/crops/object/<brand_slug>/<image_stem>_b<idx>.jpg
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelf_analysis import config, io_utils                 # noqa: E402
from shelf_analysis.classifier import CLIPClassifier         # noqa: E402
from shelf_analysis.detector import ProductDetector          # noqa: E402


def brand_slug(name: str) -> str:
    """Filesystem-safe folder name from a canonical brand label."""
    s = re.sub(r"[^A-Za-z0-9 ]+", "", name).strip().lower()
    return re.sub(r"\s+", "_", s) or "other"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed the KNN knowledge base from shelf images")
    p.add_argument("--input", required=True, help="Image file or directory")
    p.add_argument("--kb", default=str(config.KNOWLEDGE_BASE_DIR),
                   help="Destination crops/object directory")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Skip crops whose top-1 cosine similarity is below this")
    p.add_argument("--include-other", action="store_true",
                   help="Also save crops classified as 'Other' (default: skip)")
    p.add_argument("--log", default=None,
                   help="CSV log path (default: <kb>/_seed_log.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images = io_utils.list_images(args.input)
    if not images:
        print(f"No images found at {args.input!r}")
        return

    kb = Path(args.kb)
    kb.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else kb / "_seed_log.csv"

    print(f"Loading models (device: {config.DEVICE}) ...")
    detector = ProductDetector()
    classifier = CLIPClassifier()

    saved = skipped_other = skipped_score = 0
    per_brand: dict[str, int] = {}
    rows: list[dict] = []

    for img_path in images:
        print(f"\n=== {img_path.name} ===")
        image = io_utils.load_image(img_path)
        boxes = detector.detect(image)
        crops = [b.crop(image) for b in boxes]
        predictions = classifier.classify(crops)
        print(f"  detected {len(boxes)} boxes")

        for i, (box, (brand, score)) in enumerate(zip(boxes, predictions)):
            if brand == config.OTHER_LABEL and not args.include_other:
                skipped_other += 1
                continue
            if score < args.min_score:
                skipped_score += 1
                continue
            slug = brand_slug(brand)
            out_dir = kb / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{img_path.stem}_b{i:03d}.jpg"
            cv2.imwrite(str(out_dir / fname), box.crop(image))
            saved += 1
            per_brand[slug] = per_brand.get(slug, 0) + 1
            rows.append({
                "image": img_path.name, "box_idx": i,
                "brand": brand, "slug": slug, "score": f"{score:.3f}",
                "saved_as": str((out_dir / fname).relative_to(kb)),
            })

    # CSV log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "box_idx", "brand", "slug", "score", "saved_as"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Summary ---")
    print(f"saved        : {saved}")
    print(f"skipped Other: {skipped_other}")
    print(f"skipped low-score: {skipped_score}")
    print(f"log          : {log_path}")
    print(f"per-brand counts (top 15):")
    for slug, n in sorted(per_brand.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {slug:24s} {n}")


if __name__ == "__main__":
    main()
