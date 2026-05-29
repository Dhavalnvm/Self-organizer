"""CLI entrypoint for the retail shelf intelligence pipeline.

Examples
--------
    # All three test images with the CLIP zero-shot classifier (default):
    python run.py --input data/test_images --output outputs

    # A single image with the KNN fallback (DINOv2 gallery):
    python run.py --input data/test_images/img_1.jpg --classifier knn --backbone dinov2

    # Skip OCR (faster smoke run):
    python run.py --input data/test_images --no-ocr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the in-repo package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shelf_analysis import io_utils                       # noqa: E402
from shelf_analysis.classifier import build_classifier    # noqa: E402
from shelf_analysis.config import DEFAULT_OUTPUT_DIR, DEVICE  # noqa: E402
from shelf_analysis.detector import ProductDetector       # noqa: E402
from shelf_analysis.pipeline import ShelfPipeline         # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail shelf intelligence pipeline")
    p.add_argument("--input", required=True, help="Image file or directory of images")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    p.add_argument("--classifier", choices=["clip", "knn"], default="clip",
                   help="Brand classifier: clip (zero-shot, default) or knn (gallery fallback)")
    p.add_argument("--backbone", choices=["dinov2", "resnet18"], default="dinov2",
                   help="Embedding backbone for the KNN classifier")
    p.add_argument("--no-ocr", action="store_true", help="Disable the OCR stage")
    p.add_argument("--ocr-brand-correct", action="store_true",
                   help="Experimental: let confident OCR brand text override CLIP "
                        "(off by default; can hurt in brand-repeating aisles)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images = io_utils.list_images(args.input)
    if not images:
        print(f"No images found at {args.input!r}")
        return

    print(f"Device: {DEVICE} | classifier: {args.classifier} | images: {len(images)}")
    print("Loading models ...")
    detector = ProductDetector()
    classifier = build_classifier(args.classifier, backbone=args.backbone)
    ocr = None
    if not args.no_ocr:
        from shelf_analysis.ocr import ShelfOCR
        ocr = ShelfOCR()
    pipeline = ShelfPipeline(classifier=classifier, detector=detector, ocr=ocr,
                             ocr_brand_correct=args.ocr_brand_correct)

    out_dir = Path(args.output)
    for img_path in images:
        print(f"\n=== {img_path.name} ===")
        metrics, annotated = pipeline.run(img_path)

        stem = img_path.stem
        io_utils.save_json(metrics, out_dir / f"{stem}.json")
        io_utils.save_image(annotated, out_dir / f"{stem}_annotated.jpg")

        print(f"  total_products : {metrics['total_products']}")
        print(f"  shelf_rows     : {metrics['num_shelf_rows']}")
        top = list(metrics["brands"].items())[:6]
        print(f"  top brands     : {top}")
        print(f"  ocr_labels     : {metrics['ocr_labels'][:8]}")
        print(f"  -> {out_dir / (stem + '.json')}")
        print(f"  -> {out_dir / (stem + '_annotated.jpg')}")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
