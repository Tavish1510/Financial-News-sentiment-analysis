"""
Token-level explainability via attention weights.

We extract the [CLS]-token's attention to every input token in the final
transformer layer and aggregate across heads. High attention from [CLS] to
a token usually correlates with that token's contribution to the prediction.

For a stronger but more expensive approach, swap in Captum's Integrated
Gradients; the surface API is the same.
"""

from __future__ import annotations

import numpy as np
import torch

from src.predict import SentimentPredictor


def token_attributions(predictor: SentimentPredictor, text: str) -> tuple[list[str], np.ndarray, str, dict[str, float]]:
    """Return (tokens, importance, predicted_label, all_scores).

    `importance` is a 1-D array aligned with `tokens`, normalized to sum to 1
    over the actual input tokens (i.e. excluding padding).
    """
    tokenizer = predictor.tokenizer
    model = predictor.model

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(predictor.device)
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    # Final layer attentions: (batch, heads, tokens, tokens). Take CLS row.
    final_attn = outputs.attentions[-1][0]                # (heads, tokens, tokens)
    cls_attn = final_attn[:, 0, :].mean(dim=0).cpu().numpy()  # mean over heads

    # Mask out padding
    n_real = int(attention_mask.sum().item())
    cls_attn = cls_attn[:n_real]
    cls_attn = cls_attn / (cls_attn.sum() + 1e-9)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[:n_real].cpu().tolist())

    probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()
    scores = {predictor._labels[i]: float(probs[i]) for i in range(len(probs))}
    pred_label = max(scores, key=scores.get)

    return tokens, cls_attn, pred_label, scores
