"""Smoke tests for the inference wrapper.

These hit the network to download the fallback model on first run, so they're
marked as 'slow' — feel free to skip them locally with `pytest -m "not slow"`.
"""

import pytest

from src.predict import SentimentPredictor


@pytest.fixture(scope="module")
def predictor() -> SentimentPredictor:
    return SentimentPredictor()


@pytest.mark.slow
def test_single_prediction_returns_known_label(predictor):
    pred = predictor.predict("Apple beat earnings expectations by 20%.")
    assert pred.label in ("negative", "neutral", "positive")
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-3


@pytest.mark.slow
def test_batch_prediction_preserves_order(predictor):
    texts = [
        "Apple beat earnings.",
        "The company filed paperwork.",
        "Tesla missed its delivery target by a wide margin.",
    ]
    preds = predictor.predict_batch(texts)
    assert len(preds) == len(texts)
    for p, t in zip(preds, texts):
        assert p.text == t
