"""Inference wrapper for the financial sentiment model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import ID2LABEL
from src.model import load_inference_model, tokenize_batch


@dataclass
class Prediction:
    text: str
    label: str
    confidence: float
    scores: dict[str, float]


class SentimentPredictor:
    def __init__(self, prefer_finetuned: bool = True):
        self.tokenizer, self.model, self.source = load_inference_model(prefer_finetuned)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._labels = self._build_label_map()

    def _build_label_map(self) -> dict[int, str]:
        """The fallback model uses different id2label than ours — normalize."""
        cfg_labels = self.model.config.id2label
        normalized = {}
        for idx, raw in cfg_labels.items():
            r = str(raw).lower()
            if "neg" in r:
                normalized[int(idx)] = "negative"
            elif "pos" in r:
                normalized[int(idx)] = "positive"
            else:
                normalized[int(idx)] = "neutral"
        return normalized

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenize_batch(self.tokenizer, chunk).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for text, p in zip(chunk, probs):
                scores = {self._labels[i]: float(p[i]) for i in range(len(p))}
                best_label = max(scores, key=scores.get)
                results.append(
                    Prediction(
                        text=text,
                        label=best_label,
                        confidence=scores[best_label],
                        scores=scores,
                    )
                )
        return results
