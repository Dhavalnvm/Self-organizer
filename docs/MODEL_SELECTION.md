# Model Selection & Justification

All models are **pretrained and open-source**. The guiding constraint is that the
target environment is **CPU-only** (the provided venv ships `torch==2.12.0+cpu`,
CUDA unavailable), so every choice trades a little accuracy for CPU practicality.

## 1. Detection — YOLOv8m trained on SKU110K (`models/best.pt`)

**Why.** SKU110K is the canonical densely-packed retail-shelf dataset, and the
shipped weights detect individual product *facings* class-agnostically — exactly
the "how many objects are on the shelf" question. Reusing it avoids retraining and
gives strong recall on tightly packed rows.

**Alternatives considered.** A generic COCO YOLO detects "bottle"/"box" but
collapses adjacent facings and misses snack bags; an open-vocabulary detector
(GroundingDINO / YOLO-World) is heavier and slower on CPU. The SKU110K specialist
wins on both accuracy-for-task and speed.

**Accuracy ↔ speed.** YOLOv8**m** balances recall and latency. On CPU a single
shelf image runs in a few seconds; YOLOv8n would be faster but miss small facings,
YOLOv8x more accurate but too slow for CPU batch use.

## 2. Brand classification — CLIP ViT-B/32 zero-shot (primary)

**Why.** There is no labelled training set for the dozens of brands on these
shelves, and the brand list changes per retailer/category. CLIP zero-shot needs
**zero labelled data**: adding a brand is one line in `config.BRAND_PROMPTS`. We
average several text prompts per brand for robustness and route low-confidence
crops to `Other` via an **open-set raw cosine-similarity threshold** (a peaked
softmax saturates to ~1.0 and cannot reject out-of-taxonomy items; raw cosine
cleanly separates them — measured ~0.19–0.21 for unknowns vs ~0.26–0.39 for true
matches). This directly satisfies the "broad brand categories + Other" requirement
and is tuned against `data/ground_truth.json` via `scripts/evaluate.py`.

**Accuracy ↔ speed.** ViT-B/32 is the smallest CLIP that still reads packaging
text/logos well, and it encodes crops in batches — practical on CPU. ViT-L/14
would lift accuracy but is ~3× slower on CPU. Zero-shot will confuse visually
similar sub-brands (e.g. Coke vs Diet Coke); that is acceptable for a prototype
and tunable via prompts/threshold.

**Fallback — DINOv2 / ResNet18 + cosine-KNN.** When a curated gallery of labelled
crops exists (the repo's `data/knowledge_base/`), KNN over embeddings gives precise
SKU-level naming. It is selectable with `--classifier knn`. The trade-off: it only
recognises brands present in the gallery and needs manual curation — hence it is
the fallback, not the default.

## 3. OCR — EasyOCR

**Why.** Open-source, pip-installable, reuses the already-present torch/opencv, and
handles the slightly rotated, low-contrast price tags better out-of-the-box than
plain Tesseract. We OCR the whole frame; `labels.py` then keeps the standalone
price numbers (re-attaching the `₹` glyph that EasyOCR drops) as `ocr_labels`, and
ties each price to the brand column above it (`price_by_brand`).

**What works vs what doesn't.** Price extraction is reliable — the prices sit in
clean shelf-edge bands. Brand *text* on tags/packs is only partly legible, so using
it to **correct** the CLIP brand is offered but **off by default** (`--ocr-brand-
correct`): measured against `data/ground_truth.json` it was net-negative because a
dominant brand's name (e.g. "Amul" across a dairy aisle) bleeds onto neighbouring
products. This is a deliberate, measured decision — keep the reliable signal
(prices), gate the unreliable one (brand override).

**Accuracy ↔ speed.** EasyOCR's detector+recognizer is the slowest CPU stage; it
can be disabled with `--no-ocr`. PaddleOCR is comparable but adds a heavier
dependency tree; Tesseract is faster but weaker on stylised tags.

## 4. Shelf-space + Out-of-Stock — geometric (no extra model)

Row clustering (1-D gap split on box y-centres) and Share-of-Shelf (per-brand
sum of box widths) are pure geometry on the detections — zero added latency or
weights. A segmentation model (SAM) would give pixel-accurate area but is far too
slow on CPU for marginal business value here; noted as future work.

**On-Shelf Availability (`find_empty_slots`).** Same family — pure geometry on
the detections. Within each row, x-gaps between consecutive facings that exceed
**one median facing-width** are flagged as out-of-stock, with the gap size
divided by the median width giving an estimated count of missing facings. Edge
gaps need ≥ 1.5× the median width before flagging so we don't mistake normal
shelf-end whitespace for OOS.

*Why not a dedicated OOS model?* "Empty-shelf detectors" exist but they add
weights, latency, and a second labelled dataset. The gap heuristic costs
microseconds, is interpretable (you can see *why* a gap was flagged), and was
sufficient to correctly flag the depleted Coca-Cola section in `img_4` as 5
missing facings plus a smaller 2-facing gap. Failure modes: an entire empty row
disappears from `cluster_rows` (no boxes to anchor it), and stocked-but-tilted
products may briefly look like gaps.

## CPU vs GPU summary

| Concern | On CPU (this env) | On GPU |
|---------|-------------------|--------|
| Detection (YOLOv8m) | a few sec/image | tens of ms |
| CLIP encoding | batched, seconds for a full shelf | near-instant |
| EasyOCR | slowest stage; optional | fast |
| Practical mode | offline / batch analytics | real-time per-store |

**Deployment practicality.** Everything is pip-installable and runs on a laptop
CPU with no GPU — suitable for a batch analytics job or a low-QPS service. For
real-time, per-store inference, move the same code to GPU (`DEVICE` auto-detects
CUDA) and optionally swap to YOLOv8n + a quantised CLIP. The modular stage design
means each model can be upgraded independently without touching the pipeline.
