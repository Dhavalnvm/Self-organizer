"""Brand classification stage.

Two interchangeable classifiers behind one ``classify(crops)`` interface:

* ``CLIPClassifier`` (primary) — zero-shot: match each product crop against
  natural-language brand prompts with CLIP. No training or labelled data needed,
  and the brand list is just a config edit away.
* ``KNNClassifier`` (fallback) — reuses the repo's DINOv2 / ResNet18 embedding
  extractors and the labelled ``data/knowledge_base`` via cosine-distance KNN.
  Useful when a curated gallery exists and exact SKU naming matters.

Crops are passed as BGR numpy arrays (OpenCV native) and converted internally.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import config


def _bgr_to_pil(crop: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class CLIPClassifier:
    #Zero-shot brand classifier built on OpenAI CLIP (via transformers).

    def __init__(
        self,
        model_name: str = config.CLIP_MODEL_NAME,
        brand_prompts: dict[str, list[str]] = config.BRAND_PROMPTS,
        sim_threshold: float = config.CLIP_SIM_THRESHOLD,
        margin: float = config.CLIP_MARGIN,
        device: str = config.DEVICE,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.sim_threshold = sim_threshold
        self.margin = margin
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # One averaged text embedding per canonical brand.
        self.brands: list[str] = list(brand_prompts.keys())
        self.text_features = self._encode_brand_text(brand_prompts)

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        #CLIP text embeddings (pooled + projected, then L2-normalised).
        #We apply the projection head manually rather than calling
        #``get_text_features`` so the code is stable across transformers versions
        #(v5 changed that helper to return an unprojected output object).
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            pooled = self.model.text_model(**inputs).pooler_output
            emb = self.model.text_projection(pooled)
        return torch.nn.functional.normalize(emb, dim=-1)

    def _encode_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pooled = self.model.vision_model(**inputs).pooler_output
            emb = self.model.visual_projection(pooled)
        return torch.nn.functional.normalize(emb, dim=-1)

    def _encode_brand_text(self, brand_prompts: dict[str, list[str]]) -> torch.Tensor:
        per_brand = []
        for prompts in brand_prompts.values():
            texts = [config.PROMPT_TEMPLATE.format(name=p) for p in prompts]
            feats = self._encode_text(texts)
            per_brand.append(feats.mean(dim=0))          # average aliases
        stacked = torch.stack(per_brand, dim=0)
        return torch.nn.functional.normalize(stacked, dim=-1)

    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        pil_images = [_bgr_to_pil(c) for c in crops]
        results: list[tuple[str, float]] = []
        batch = 64
        for i in range(0, len(pil_images), batch):
            chunk = pil_images[i : i + batch]
            img_feats = self._encode_images(chunk)
            # Open-set decision on the RAW cosine similarity (both sides are L2
            # normalised, so this is cosine). A crop is "Other" when its best
            # brand match is too weak, or when the top two brands are too close.
            sims = img_feats @ self.text_features.T            # (N, B)
            top2 = torch.topk(sims, k=min(2, sims.shape[1]), dim=-1)
            best_sim = top2.values[:, 0]
            best_idx = top2.indices[:, 0]
            second_sim = top2.values[:, 1] if sims.shape[1] > 1 else best_sim
            for s1, s2, idx in zip(best_sim.tolist(), second_sim.tolist(), best_idx.tolist()):
                if s1 < self.sim_threshold or (s1 - s2) < self.margin:
                    results.append((config.OTHER_LABEL, float(s1)))
                else:
                    results.append((self.brands[idx], float(s1)))
        return results


class KNNClassifier:
    #Cosine-KNN over the repo's embedding gallery (DINOv2 / ResNet18).

    def __init__(
        self,
        backbone: str = config.KNN_DEFAULT_BACKBONE,
        knowledge_base_dir: str | Path = config.KNOWLEDGE_BASE_DIR,
        n_neighbors: int = config.KNN_N_NEIGHBORS,
    ) -> None:
        from sklearn.neighbors import NearestNeighbors

        self.backbone = backbone
        self.n_neighbors = n_neighbors
        self._embed = self._build_embedder(backbone)

        # Fit KNN on the labelled knowledge-base crops.
        paths = sorted(Path(knowledge_base_dir).glob("**/*.jpg"))
        if not paths:
            raise FileNotFoundError(
                f"No knowledge-base crops found under {knowledge_base_dir}. "
                "The KNN fallback needs a labelled gallery."
            )
        self.labels: list[str] = []
        embeddings: list[np.ndarray] = []
        for p in paths:
            with Image.open(p) as im:
                embeddings.append(self._embed(im.convert("RGB")))
            self.labels.append(p.parent.name)
        k = min(self.n_neighbors, len(embeddings))
        self.knn = NearestNeighbors(metric="cosine", n_neighbors=k).fit(embeddings)

    def _build_embedder(self, backbone: str):
        #Reuse the repo's Img2Vec* classes; return a PIL->vector callable.
        src_dir = config.REPO_ROOT / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        if backbone == "dinov2":
            from img2vec_dino2 import Img2VecDino2

            m = Img2VecDino2()

            def embed(pil: Image.Image) -> np.ndarray:
                # Call the loaded model directly so we accept a PIL image
                # (the repo's getVec opens a path; we reuse the weights, not it).
                inputs = m.processor(images=pil, return_tensors="pt").to(m.device)
                with torch.no_grad():
                    out = m.model(**inputs)
                emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], dim=-1)
                return emb.squeeze().cpu().numpy()

            return embed

        from img2vec_resnet18 import Img2VecResnet18

        m = Img2VecResnet18()
        return lambda pil: m.getVec(pil)

    def classify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for crop in crops:
            vec = self._embed(_bgr_to_pil(crop))
            _, idx = self.knn.kneighbors([vec])
            neighbours = [self.labels[i] for i in idx[0]]
            count = Counter(neighbours)
            label, n = count.most_common(1)[0]
            results.append((label, n / len(neighbours)))
        return results


def build_classifier(kind: str = "clip", backbone: str = config.KNN_DEFAULT_BACKBONE):
    """Factory: ``kind`` is "clip" (primary) or "knn" (fallback)."""
    if kind == "clip":
        return CLIPClassifier()
    if kind == "knn":
        return KNNClassifier(backbone=backbone)
    raise ValueError(f"Unknown classifier kind: {kind!r} (use 'clip' or 'knn').")
