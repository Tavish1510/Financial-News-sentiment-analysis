"""Model and tokenizer loading utilities."""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import BASE_MODEL, FALLBACK_MODEL, HF_HUB_REPO, ID2LABEL, LABEL2ID


def load_base_model_for_training(
    model_name: str = BASE_MODEL,
    num_labels: int = 3,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load tokenizer + base model from HuggingFace ready for fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model


def load_inference_model(
    prefer_finetuned: bool = True,
) -> tuple[PreTrainedTokenizer, PreTrainedModel, str]:
    """Load the best available model for inference.

    Tries the fine-tuned model on HF Hub first, falls back to the public
    pre-trained financial sentiment model if that isn't available.

    Returns:
        (tokenizer, model, source_label)
    """
    if prefer_finetuned:
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_HUB_REPO)
            model = AutoModelForSequenceClassification.from_pretrained(HF_HUB_REPO)
            model.eval()
            return tokenizer, model, f"fine-tuned ({HF_HUB_REPO})"
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
    model.eval()
    return tokenizer, model, f"fallback ({FALLBACK_MODEL})"


def tokenize_batch(tokenizer: PreTrainedTokenizer, texts: list[str], max_length: int = 128):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
