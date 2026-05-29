# Retail Shelf Intelligence Pipeline

A prototype ML pipeline that analyzes retail shelf images and produces **brand-wise
shelf presence + product availability** insights. For each shelf image it:

1. **Detects** product facings (YOLOv8 trained on SKU110K)
2. **Classifies** each facing into a brand (CLIP zero-shot; embedding+KNN fallback)
3. **OCRs** shelf labels / price tags (EasyOCR)
4. **Estimates shelf space** — groups facings into rows and computes Share-of-Shelf
5. **Outputs** a business-metrics JSON + an annotated image

Built on top of the original *facings identifier* repo (a YOLOv8/SKU110K detector
plus DINOv2/ResNet18 image embeddings), extended into a clean, modular pipeline.

> This is a prototype, optimised for **practicality on CPU**, not production
> perfection. See [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) for the
> accuracy/speed and CPU-vs-GPU reasoning, and [Limitations](#assumptions--limitations).

## Architecture

```
image ─▶ Detection (YOLOv8/SKU110K) ─▶ crop facings ─▶ Brand classifier
                                                          ├─ CLIP zero-shot (primary)
                                                          └─ DINOv2/ResNet18 + KNN (fallback)
image ─▶ OCR (EasyOCR) ─────────────────────────────────────────────┐
detections ─▶ Shelf-space (rows + Share-of-Shelf) ──────────────────┤
                                                                     ▼
                                                  metrics JSON + annotated image
```

Full diagram and stage table: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

Requires Python 3.10+. A CPU-only PyTorch is fine (the project is CPU-friendly).

```bash
# from the repo root
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

`models/best.pt` (the SKU110K YOLOv8 detector) ships with the repo. CLIP, DINOv2
and EasyOCR weights download automatically on first run.

## Test images

The three provided shelf images live in `data/test_images/`:

```
data/test_images/
  img_1.jpg   # beverages (juices, sodas, energy drinks)
  img_2.jpg   # snacks / biscuits
  img_3.jpg   # dairy (milk, yoghurt, butter, cheese)
```

## Usage

```bash
# Analyze all three test images (CLIP classifier, OCR on):
python run.py --input data/test_images --output outputs

# A single image:
python run.py --input data/test_images/img_1.jpg

# KNN fallback over the labelled knowledge base (DINOv2 backbone):
python run.py --input data/test_images/img_1.jpg --classifier knn --backbone dinov2

# Faster run without OCR:
python run.py --input data/test_images --no-ocr

# Experimental: let confident OCR brand text override CLIP (see note below):
python run.py --input data/test_images --ocr-brand-correct
```

For each input image the pipeline writes:

- `outputs/<name>.json` — the metrics dict
- `outputs/<name>_annotated.jpg` — product boxes + brand labels + price-tag overlays

### Output schema

```json
{
  "image_name": "img_1.jpg",
  "total_products": 83,
  "brands": { "Minute Maid": 7, "Tropicana": 5, "Coca-Cola": 4, "Other": 4 },
  "share_of_shelf": { "Minute Maid": "8.5%", "Tropicana": "6.0%" },
  "num_shelf_rows": 4,
  "ocr_labels": ["₹30", "₹50", "₹99", "₹125"],
  "price_by_brand": { "Coca-Cola": "₹50", "Gatorade": "₹75", "Red Bull": "₹110" }
}
```

`image_name`, `total_products`, `brands`, `ocr_labels` satisfy the assignment's
required schema; `share_of_shelf`, `num_shelf_rows` and `price_by_brand` are added
insights. `ocr_labels` are the price tags read off the shelf edges (the `₹` glyph
is re-attached — EasyOCR drops it); `price_by_brand` ties each price to the brand
of the products directly above its tag.

> **Note on `--ocr-brand-correct`:** using OCR'd shelf/pack text to override the
> CLIP brand is *experimental and off by default*. Measured against
> `data/ground_truth.json` it was **net-negative** here: in aisles dominated by one
> brand (e.g. "Amul" repeated across every dairy tag) the brand name bleeds into
> neighbouring products. Kept as an opt-in for clean, non-repeating labels.

## Project layout

```
src/
  shelf_analysis/
    config.py       # paths, device, thresholds, BRAND_PROMPTS taxonomy
    detector.py     # ProductDetector (YOLOv8 wrapper) + Box
    classifier.py   # CLIPClassifier (primary) + KNNClassifier (fallback)
    ocr.py          # ShelfOCR (EasyOCR text detection)
    labels.py       # price extraction, price-by-brand, OCR brand matching
    shelf_space.py  # row clustering + Share-of-Shelf
    visualize.py    # annotated-image rendering
    pipeline.py     # ShelfPipeline orchestration
    io_utils.py     # image / JSON I/O
  img2vec_dino2.py    # DINOv2 embedder  (reused by the KNN fallback)
  img2vec_resnet18.py # ResNet18 embedder (reused by the KNN fallback)
run.py              # CLI entrypoint
scripts/            # evaluate.py, make_diagram.py
docs/               # ARCHITECTURE.md, MODEL_SELECTION.md, architecture.png
data/               # test_images/, knowledge_base/ (KNN gallery), ground_truth.json
models/best.pt      # YOLOv8 SKU110K detector
outputs/            # generated JSON + annotated images
```

## Configuring brands

Brand recognition is driven entirely by `BRAND_PROMPTS` in
[src/shelf_analysis/config.py](src/shelf_analysis/config.py): canonical brand →
list of text prompts. Add or remove a brand by editing this dict — no retraining.
`CLIP_SIM_THRESHOLD` is the open-set cutoff: a crop whose best brand cosine
similarity falls below it is labelled `Other` (this rejects out-of-taxonomy items
instead of forcing them onto the nearest brand). `CLIP_MARGIN` optionally rejects
crops where the top-2 brands are too close.

## Evaluation

Manual facing counts live in [data/ground_truth.json](data/ground_truth.json).
Score the current outputs against them:

```bash
python scripts/evaluate.py --outputs outputs --gt data/ground_truth.json
```

It prints per-image, per-brand absolute count error and an overall MAE, so any
prompt/threshold change is measurable. Flags false positives (`pred>0, true=0`)
and misses (`pred=0, true>0`).

## Assumptions & limitations

- **Detector is class-agnostic** (SKU110K finds product *facings*; the classifier
  names them). Tightly packed or occluded items may merge/split.
- **CLIP zero-shot** captures total counts well but can split near-identical
  packages between lookalike brands (e.g. Tropicana / Real / Minute Maid juice
  cartons). Tune `BRAND_PROMPTS` / `CLIP_SIM_THRESHOLD`, or use the `--classifier
  knn` gallery path for logo-level discrimination. No labelled data required.
- **Share-of-Shelf is a bbox-width proxy** for linear frontage, not pixel-accurate
  segmentation (SAM noted as future work in MODEL_SELECTION.md).
- **OCR**: price tags read reliably (`ocr_labels`, `price_by_brand`). Brand *text*
  on tags/packs is only partly legible, so OCR-based brand **correction** is
  experimental/off by default (it bleeds in brand-repeating aisles — see Usage).
- **Lookalike packages**: near-identical SKUs (e.g. Tropicana / Real / Minute Maid
  juice cartons) get the right *total* but are split imperfectly between the three;
  reliable separation needs the KNN gallery path or a higher-resolution model.
- **CPU-only**: first run downloads model weights; per-image latency is seconds,
  suitable for batch analytics. `DEVICE` auto-detects CUDA for GPU deployment.
