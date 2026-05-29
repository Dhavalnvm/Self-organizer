"""Compare pipeline outputs against manual ground truth.

    python scripts/evaluate.py --outputs outputs --gt data/ground_truth.json

Reports, per image: detected vs true total, and per-brand absolute count error
(|pred - true|). Prints an overall count-MAE so tuning changes are measurable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--gt", default="data/ground_truth.json")
    args = ap.parse_args()

    gt = load(args.gt)
    out_dir = Path(args.outputs)

    grand_abs = 0
    grand_brands = 0
    for name, truth in gt.items():
        if name.startswith("_"):
            continue
        stem = Path(name).stem
        pred_path = out_dir / f"{stem}.json"
        if not pred_path.exists():
            print(f"[skip] no output for {name}")
            continue
        pred = load(pred_path)
        pb = pred.get("brands", {})
        tb = truth["brands"]

        print(f"\n=== {name} ===")
        print(f"  total: pred={pred['total_products']}  true={truth['total_products']}"
              f"  (diff {pred['total_products'] - truth['total_products']:+d})")

        brands = sorted(set(pb) | set(tb))
        img_abs = 0
        print(f"  {'brand':16s} {'pred':>4s} {'true':>4s} {'|err|':>5s}")
        for b in brands:
            p, t = pb.get(b, 0), tb.get(b, 0)
            err = abs(p - t)
            img_abs += err
            flag = "  <-- FP" if t == 0 and p > 0 else ("  <-- miss" if p == 0 and t > 0 else "")
            print(f"  {b:16s} {p:>4d} {t:>4d} {err:>5d}{flag}")
        mae = img_abs / max(1, len(brands))
        print(f"  per-brand count MAE: {mae:.2f}  (sum |err| = {img_abs})")
        grand_abs += img_abs
        grand_brands += len(brands)

    print(f"\nOVERALL per-brand count MAE: {grand_abs / max(1, grand_brands):.3f}"
          f"  (total sum |err| = {grand_abs})")


if __name__ == "__main__":
    main()
