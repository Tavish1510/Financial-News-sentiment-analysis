# Model Card — DistilBERT Financial PhraseBank

## Model details

- **Architecture:** DistilBERT-base-uncased fine-tuned for sequence classification
- **Parameters:** ~67M
- **Classes:** 3 — `negative`, `neutral`, `positive`
- **Tokenizer:** DistilBERT WordPiece, max_length = 128
- **Author:** Tavish Tayal
- **HuggingFace Hub:** [`Tavish15100/distilbert-financial-phrasebank`](https://huggingface.co/Tavish15100/distilbert-financial-phrasebank)

## Training data

**Financial PhraseBank** (Malo et al., 2014) — a dataset of 4,840 sentences from financial news, hand-annotated by domain experts.

- Configuration used: `sentences_75agree` (3,453 sentences with ≥75% inter-annotator agreement)
- Class distribution:
  - Neutral ~60%
  - Positive ~28%
  - Negative ~12%

Split into train / validation / test = 80 / 10 / 10 with stratified sampling on label.

## Training procedure

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Epochs | 4 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Optimizer | AdamW |
| Max sequence length | 128 |
| Mixed precision | fp16 on GPU |
| Hardware | NVIDIA T4 (Google Colab) |
| Training time | ~5 minutes |

## Evaluation

Held-out test set (10%, stratified, n = 345). Full breakdown in `notebooks/03_evaluation.ipynb`.

| Metric           | DistilBERT (fine-tuned) | TF-IDF + Logistic Regression baseline |
|------------------|--------------------------|----------------------------------------|
| Accuracy         | 0.9217                   | 0.7710                                 |
| F1 macro         | 0.9084                   | 0.7305                                 |
| F1 weighted      | 0.9222                   | 0.7698                                 |
| Precision macro  | 0.9140                   | —                                      |
| Recall macro     | 0.9042                   | —                                      |

Macro F1 is the headline metric since the neutral class dominates (~62% of test) — weighted F1 would over-reward correct predictions on the majority class.

## Intended use

- **In-scope:** Sentiment classification of English-language financial news *sentences* and headlines. Designed for research and demo purposes.
- **Out-of-scope:** Trading decisions, long-form documents (>128 tokens), non-English text, social media slang.

## Limitations

- **Domain shift:** Trained on news from the 2010s. Modern jargon (e.g. crypto, meme stocks) may be misclassified.
- **Class imbalance:** Neutral over-represented; the model is more conservative than it should be on truly positive/negative headlines.
- **No causal claim:** Sentiment ≠ price impact. Don't use this as a trading signal.

## Ethical considerations

- The training data is in English and primarily about US/EU equities, which biases the model towards Western financial language and may misclassify news about emerging markets.
- The model can produce confident wrong answers. Always show the confidence score and don't auto-trade on it.

## Citation

```bibtex
@article{malo2014good,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal={Journal of the Association for Information Science and Technology},
  volume={65},
  number={4},
  pages={782--796},
  year={2014},
}
```
