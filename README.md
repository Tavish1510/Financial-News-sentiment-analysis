# Financial News Sentiment Analysis

DistilBERT fine-tuned on the Financial PhraseBank for 3-class sentiment classification (negative / neutral / positive), served through a multi-page Streamlit app.

**Model:** [Tavish15100/distilbert-financial-phrasebank](https://huggingface.co/Tavish15100/distilbert-financial-phrasebank) · **Demo:** [financial-news-sentiment-analysis-tavish.streamlit.app](https://financial-news-sentiment-analysis-tavish.streamlit.app/)

## Results

10% held-out test split (n = 345):

| Metric            | TF-IDF + LR baseline | DistilBERT (fine-tuned) |
|-------------------|----------------------|--------------------------|
| Accuracy          | 0.7710               | **0.9217**               |
| F1 macro          | 0.7305               | **0.9084**               |
| F1 weighted       | 0.7698               | **0.9222**               |
| Precision (macro) | —                    | 0.9140                   |
| Recall (macro)    | —                    | 0.9042                   |

See `notebooks/03_evaluation.ipynb` for the per-class breakdown, confusion matrix, and error analysis.

## App

Four pages:

- **Single headline** — paste a sentence, get sentiment + class probabilities
- **Batch analysis** — paste many headlines or upload CSV, returns distribution stats and downloadable scored CSV
- **Live news** — pulls headlines from Reuters, MarketWatch, Yahoo Finance, CNBC and Investing.com RSS feeds, scores in real time
- **Explainability** — token-level attention attribution (which tokens drove the prediction)

## Stack

PyTorch · HuggingFace Transformers · Datasets · scikit-learn · Streamlit · Plotly · feedparser · GitHub Actions (ruff + pytest)

## Project layout

```
src/
  config.py          # central config (model, hyperparams, RSS feeds)
  data.py            # Financial PhraseBank loader (downloads from HF mirror)
  model.py           # tokenizer + model loading with fallback
  train.py           # Trainer-based fine-tuning pipeline
  predict.py         # SentimentPredictor batch-inference wrapper
  explainability.py  # attention-based token attribution
  news_feed.py       # RSS feed parser
notebooks/
  01_exploration.ipynb    # EDA: class balance, length, vocab
  02_fine_tuning.ipynb    # Colab-ready training
  03_evaluation.ipynb     # test metrics + confusion matrix + baseline
streamlit_app/
  app.py
  pages/                  # 4 Streamlit pages
reports/
  model_card.md
```

## Quick start

```bash
git clone https://github.com/Tavish1510/Financial-News-sentiment-analysis.git
cd Financial-News-sentiment-analysis
python -m venv .venv && source .venv/bin/activate     # or .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

To retrain: open `notebooks/02_fine_tuning.ipynb` in Google Colab with a T4 GPU and run all cells. The notebook pushes the trained weights to your HuggingFace Hub repo. Update `HF_HUB_REPO` in `src/config.py` if you use a different repo name.

## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014) — 4,840 sentences from financial news, hand-annotated by domain experts. This project uses the `sentences_75agree` subset (≥75% inter-annotator agreement, 3,453 sentences).
