"""Central configuration: paths, device, thresholds and the brand taxonomy.

Everything tunable lives here so the pipeline code stays declarative. The brand
taxonomy is a *rich* set chosen to cover the brands actually visible in the three
provided test images (dairy, snacks/biscuits, beverages) plus an implicit "Other"
bucket for anything the classifier is not confident about.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repo root: <repo>/src/shelf_analysis/config.py)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "best.pt"
KNOWLEDGE_BASE_DIR = REPO_ROOT / "data" / "knowledge_base" / "crops" / "object"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Device — resolves to "cpu" on the provided venv (torch CPU build).
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
DETECT_CONF = 0.35          # YOLO confidence threshold (raised to cut duplicate/partial boxes)
DETECT_IOU = 0.45           # NMS IoU (slightly tighter to merge split facings)
DETECT_MAX_DET = 1000       # plenty for a packed shelf

# --------------------------------------------------------------------------- #
# CLIP classifier
# --------------------------------------------------------------------------- #
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OTHER_LABEL = "Other"
# Open-set rejection on the RAW cosine similarity (not a peaked softmax, which
# saturated to ~1.0 and made the threshold useless). Calibrated from the sim
# distribution on the test images: correct matches score ~0.26-0.39, while
# out-of-taxonomy crops fall to ~0.19-0.21. A crop whose best brand similarity is
# below this is labelled "Other".
CLIP_SIM_THRESHOLD = 0.235
# Optional ambiguity guard: if the top-1 and top-2 brands are within this cosine
# margin the crop is "Other". 0.0 disables it (close pairs like Coke/Diet Coke are
# legitimately near, so keep this small or off).
CLIP_MARGIN = 0.0

# Prompt template used to turn each brand name into a CLIP text query.
PROMPT_TEMPLATE = "a product package of {name} on a retail store shelf"

# Canonical brand -> list of natural-language aliases fed to CLIP. Multiple
# prompts per brand are averaged, which makes zero-shot matching more robust.
BRAND_PROMPTS: dict[str, list[str]] = {
    # ---- Beverages: Coca-Cola company ----
    "Coca-Cola": ["Coca-Cola", "Coca Cola soft drink", "Coke"],
    "Diet Coke": ["Diet Coke"],
    "Thums Up": ["Thums Up cola"],
    "Sprite": ["Sprite lemon soda"],
    "Fanta": ["Fanta orange soda"],
    "Limca": ["Limca lemon soda"],
    "Minute Maid": ["Minute Maid fruit juice carton tetra pack"],
    # ---- Beverages: PepsiCo ----
    "Pepsi": ["Pepsi cola"],
    "Mountain Dew": ["Mountain Dew soda"],
    "Mirinda": ["Mirinda orange soda"],
    "7UP": ["7UP lemon soda", "Seven Up"],
    "Tropicana": ["Tropicana fruit juice carton tetra pack"],
    "Gatorade": ["Gatorade sports drink bottle"],
    # ---- Beverages: other ----
    "Real": ["Real fruit juice carton tetra pack"],
    "Paper Boat": ["Paper Boat juice drink bottle"],
    "B Natural": ["B Natural fruit juice bottle"],
    "Red Bull": ["Red Bull energy drink slim can"],
    "Nescafe": ["Nescafe cold coffee bottle"],
    "Lipton": ["Lipton iced tea bottle"],
    "Nestea": ["Nestea iced tea bottle"],
    # ---- Snacks: PepsiCo / Frito-Lay ----
    "Lay's": ["Lay's potato chips", "Lays chips"],
    "Doritos": ["Doritos tortilla chips"],
    "Cheetos": ["Cheetos snack"],
    "Kurkure": ["Kurkure snack"],
    "Uncle Chipps": ["Uncle Chipps potato chips"],
    # ---- Snacks: other ----
    "Bingo": ["Bingo snack chips"],
    "Pringles": ["Pringles potato crisps can"],
    # ---- Biscuits ----
    "Oreo": ["Oreo cookies"],
    "Parle": ["Parle-G biscuits", "Parle Monaco biscuits"],
    "Good Day": ["Britannia Good Day cookies"],
    "Britannia": ["Britannia biscuits", "Britannia Marie Gold", "Britannia Tiger"],
    "Hide & Seek": ["Hide and Seek chocolate biscuits"],
    "Dark Fantasy": ["Sunfeast Dark Fantasy cookies"],
    # ---- Dairy ----
    "Amul": ["Amul dairy product", "Amul milk", "Amul butter", "Amul cheese"],
    "Nestle": ["Nestle dairy product", "Nestle a+ yoghurt", "Nestle a+ dahi"],
    "Mother Dairy": ["Mother Dairy milk"],
    "Danone": ["Danone Actimel yoghurt drink", "Activia yoghurt"],
    "Yakult": ["Yakult probiotic drink"],
    "Epigamia": ["Epigamia milk shake yoghurt"],
    "Milky Mist": ["Milky Mist dairy product"],
    "Hershey's": ["Hershey's milkshake chocolate syrup"],
}

# Optional grouping of fine-grained brands into the assignment's broad parent
# brands. Kept available for reporting but not applied by default (we report the
# rich per-brand labels). Map canonical -> parent company.
BRAND_PARENT = {
    "Coca-Cola": "Coca-Cola", "Diet Coke": "Coca-Cola", "Thums Up": "Coca-Cola",
    "Sprite": "Coca-Cola", "Fanta": "Coca-Cola", "Limca": "Coca-Cola",
    "Minute Maid": "Coca-Cola",
    "Pepsi": "PepsiCo", "Mountain Dew": "PepsiCo", "Mirinda": "PepsiCo",
    "7UP": "PepsiCo", "Tropicana": "PepsiCo", "Gatorade": "PepsiCo",
    "Lay's": "PepsiCo", "Doritos": "PepsiCo", "Cheetos": "PepsiCo",
    "Kurkure": "PepsiCo", "Uncle Chipps": "PepsiCo",
}

# --------------------------------------------------------------------------- #
# KNN fallback classifier
# --------------------------------------------------------------------------- #
KNN_N_NEIGHBORS = 5
KNN_DEFAULT_BACKBONE = "dinov2"   # "dinov2" | "resnet18"

# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
OCR_LANGS = ["en"]
OCR_MIN_CONF = 0.30
CURRENCY = "₹"   # EasyOCR drops the glyph; we re-attach it to detected prices.
# A standalone price on a shelf tag: 1-4 digits, optionally with the rupee/Rs prefix.
PRICE_REGEX = re.compile(r"^\s*(?:₹|rs\.?)?\s*(\d{1,4})\s*$", re.IGNORECASE)
# Plausible product price range (rupees) — filters stray numbers like weights/years.
PRICE_MIN, PRICE_MAX = 5, 2000

# OCR-assisted brand resolution: when readable on-pack / shelf-tag text confidently
# matches a known brand, override the (weaker) CLIP guess. EXPERIMENTAL and OFF by
# default: measured net-negative on these images because a dominant brand's name
# (e.g. "Amul" across a dairy aisle) bleeds into neighbouring products. Enable with
# run.py --ocr-brand-correct only where labels are clean and brands don't repeat.
OCR_BRAND_CORRECTION = False
OCR_BRAND_MIN_CONF = 0.55     # EasyOCR confidence floor for a token to vote
OCR_BRAND_MIN_RATIO = 0.82    # fuzzy similarity floor (difflib ratio) to a brand keyword
OCR_OVERRIDE_MIN_SCORE = 0.55  # conf*ratio floor to override the CLIP label
# Generic packaging words to ignore when deriving brand keywords from the prompts.
OCR_STOPWORDS = {
    "the", "and", "for", "with", "soda", "cola", "juice", "drink", "bottle", "can",
    "carton", "tetra", "pack", "energy", "fruit", "iced", "ice", "tea", "milk",
    "shake", "coffee", "cold", "yoghurt", "yogurt", "dahi", "dairy", "product",
    "cookies", "biscuits", "biscuit", "chocolate", "chips", "crisps", "snack",
    "potato", "tortilla", "sports", "orange", "lemon", "store", "shelf", "retail",
    "slim", "salted", "probiotic", "package",
}

# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
BOX_COLOR = (0, 200, 0)
OCR_COLOR = (0, 140, 255)
TEXT_COLOR = (255, 255, 255)
