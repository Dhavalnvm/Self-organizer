# Pipeline Architecture

The pipeline is a linear sequence of independent, swappable stages. Each stage is
a small module under [`src/shelf_analysis/`](../src/shelf_analysis) so it can be
tested and replaced in isolation.

```mermaid
flowchart TD
    A[Shelf image] --> B[Detection<br/>YOLOv8 - best.pt SKU110K]
    B -->|product boxes| C[Crop each facing]
    C --> D{Brand classifier}
    D -->|primary| D1[CLIP zero-shot<br/>clip-vit-base-patch32]
    D -->|fallback| D2[Embedding + KNN<br/>DINOv2 / ResNet18 + knowledge_base]
    A --> E[OCR<br/>EasyOCR text detection]
    E --> E2[Shelf labels - labels.py<br/>prices + price-by-brand]
    B --> F[Shelf-space<br/>row clustering + Share-of-Shelf]
    D1 --> G
    D2 --> G[Aggregate metrics]
    E2 --> G
    F --> G
    G --> H[(metrics JSON)]
    G --> I[Annotated image<br/>boxes + brands + price tags]
```

## Stages

| Stage | Module | Model / method | Output |
|-------|--------|----------------|--------|
| Detection | `detector.py` | YOLOv8m, `models/best.pt` (SKU110K) | class-agnostic product boxes |
| Classification | `classifier.py` | CLIP ViT-B/32 zero-shot (primary); DINOv2/ResNet18 + cosine-KNN (fallback) | brand label + score per box |
| OCR | `ocr.py` | EasyOCR (`en`) | raw text detections + polygons |
| Shelf labels | `labels.py` | price regex + currency; column geometry; fuzzy brand match | `ocr_labels` (prices), `price_by_brand`; optional brand override |
| Shelf-space | `shelf_space.py` | 1-D gap clustering on box y-centres; bbox-width share | `num_shelf_rows`, `share_of_shelf` |
| Aggregation | `pipeline.py` | — | metrics dict |
| Visualization | `visualize.py` | OpenCV drawing | annotated JPG (boxes + brands + price tags) |

## Data flow

1. **Detect** — `ProductDetector.detect(image)` returns a list of `Box(xyxy, conf)`.
   The detector is class-agnostic: it finds *where* products are, not *what* they are.
2. **Classify** — each box is cropped and passed to the active `BrandClassifier`.
   CLIP compares the crop's embedding against averaged text embeddings of each
   brand's prompts; the arg-max brand wins, or `Other` if confidence is low.
3. **OCR + labels** — EasyOCR reads the whole image (price tags sit on shelf-edge
   strips); `labels.py` keeps the standalone price numbers (re-attaching `₹`) as
   `ocr_labels`, and ties each price to the brand of the products above it
   (`price_by_brand`). Optionally (off by default) it can override the CLIP brand
   when on-pack/tag text confidently names a known brand — see MODEL_SELECTION.md.
4. **Shelf-space** — boxes are clustered into rows by vertical position, and each
   brand's share of total box width gives Share-of-Shelf (SOS).
5. **Aggregate & visualize** — counts, SOS, rows, prices are assembled into the
   metrics dict; the annotated image overlays product boxes, brand labels and the
   price tags.

## Output schema

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

`image_name`, `total_products`, `brands` and `ocr_labels` satisfy the assignment's
required schema; `share_of_shelf`, `num_shelf_rows` and `price_by_brand` are added
insights.

A static PNG version of the diagram can be regenerated with
[`scripts/make_diagram.py`](../scripts/make_diagram.py).
