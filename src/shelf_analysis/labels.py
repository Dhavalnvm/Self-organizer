"""Shelf-label understanding from OCR.

Two jobs, both driven by the EasyOCR detections (``OCRItem``s):

1. **Prices** — pull standalone price numbers off the shelf-edge tags, re-attach
   the currency symbol, and tie each price to the brand column above it.
2. **Brand resolution** — when readable on-pack or tag text confidently matches a
   known brand, vote that brand for the products in that column. This corrects
   CLIP on visually-identical packages whose brand is legible (e.g. the Tropicana
   juice cartons that CLIP confuses with Minute Maid).

Everything is spatial + fuzzy-string; no extra model or dependency (difflib is
stdlib). Matching is deliberately conservative so garbled text falls back to CLIP.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from . import config
from .detector import Box
from .ocr import OCRItem

_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _ALNUM.sub("", text.lower())


def build_brand_keywords(
    brand_prompts: dict[str, list[str]] = config.BRAND_PROMPTS,
    stopwords: set[str] = config.OCR_STOPWORDS,
) -> dict[str, list[str]]:
    # Derive distinctive lowercase keywords per brand from its name + prompts.
    # e.g. "Minute Maid" -> ["minute", "maid", "minutemaid"]; generic packaging
    # words ("juice", "carton", ...) are dropped so they don't cause cross-matches.
    
    keywords: dict[str, list[str]] = {}
    for brand, prompts in brand_prompts.items():
        words: set[str] = set()
        for phrase in [brand, *prompts]:
            for tok in re.split(r"[^a-z0-9]+", phrase.lower()):
                if len(tok) >= 3 and tok not in stopwords:
                    words.add(tok)
        words.add(_norm(brand))           # compact form: "cocacola", "dietcoke"
        keywords[brand] = sorted(w for w in words if len(w) >= 3)
    return keywords


_BRAND_KEYWORDS = build_brand_keywords()


def match_brand(text: str) -> tuple[str | None, float]:
    #Best brand match for one OCR token. Returns (brand|None, ratio[0..1]).
    tok = _norm(text)
    if len(tok) < 3:
        return None, 0.0
    best_brand, best_ratio = None, 0.0
    for brand, keys in _BRAND_KEYWORDS.items():
        for kw in keys:
            # Substring hit on a reasonably long keyword is a strong signal.
            if len(kw) >= 4 and (kw in tok or tok in kw):
                ratio = 0.95
            else:
                ratio = SequenceMatcher(None, tok, kw).ratio()
            if ratio > best_ratio:
                best_brand, best_ratio = brand, ratio
    return best_brand, best_ratio


def extract_prices(items: list[OCRItem]) -> list[dict]:
    #Standalone price tokens -> [{'price': int, 'text': '₹125', 'cx', 'cy'}].
    out: list[dict] = []
    for it in items:
        m = config.PRICE_REGEX.match(it.text)
        if not m:
            continue
        value = int(m.group(1))
        if not (config.PRICE_MIN <= value <= config.PRICE_MAX):
            continue
        cx = sum(p[0] for p in it.box) / 4.0
        cy = sum(p[1] for p in it.box) / 4.0
        out.append({"price": value, "text": f"{config.CURRENCY}{value}", "cx": cx, "cy": cy})
    return out


def ocr_brand_for_boxes(boxes: list[Box], items: list[OCRItem]) -> list[str | None]:
    # For each box, a confident OCR-voted brand (or None to keep CLIP).
    # Considers OCR tokens whose centre falls within the box's horizontal span and
    # within the box vertically or just below it (the shelf tag). Votes are weighted
    # by EasyOCR confidence x fuzzy ratio; the best vote wins if it clears the floor.
    
    votes: list[str | None] = [None] * len(boxes)
    for bi, box in enumerate(boxes):
        x1, x2 = box.x1, box.x2
        y_lo, y_hi = box.y1, box.y2 + 0.6 * box.height   # box + tag strip below
        best_brand, best_score = None, 0.0
        for it in items:
            if it.conf < config.OCR_BRAND_MIN_CONF:
                continue
            cx = sum(p[0] for p in it.box) / 4.0
            cy = sum(p[1] for p in it.box) / 4.0
            if not (x1 <= cx <= x2 and y_lo <= cy <= y_hi):
                continue
            brand, ratio = match_brand(it.text)
            if brand is None or ratio < config.OCR_BRAND_MIN_RATIO:
                continue
            score = it.conf * ratio
            if score > best_score:
                best_brand, best_score = brand, score
        if best_brand is not None and best_score >= config.OCR_OVERRIDE_MIN_SCORE:
            votes[bi] = best_brand
    return votes


def prices_by_brand(
    prices: list[dict], boxes: list[Box], brands: list[str]
) -> dict[str, str]:
    #Associate each price tag with the brand of the nearest product above it.
    result: dict[str, str] = {}
    for pr in prices:
        cx, cy = pr["cx"], pr["cy"]
        best_bi, best_dy = None, None
        for bi, box in enumerate(boxes):
            if box.x1 <= cx <= box.x2 and box.y2 <= cy:       # column, product above tag
                dy = cy - box.y2
                if best_dy is None or dy < best_dy:
                    best_bi, best_dy = bi, dy
        if best_bi is not None:
            brand = brands[best_bi]
            result.setdefault(brand, pr["text"])              # first/closest price per brand
    return result
