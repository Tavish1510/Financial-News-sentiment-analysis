# Financial News Sentiment Analysis

Fine-tuned **DistilBERT** for 3-class sentiment classification (negative / neutral / positive) of financial news. The model is fine-tuned on the **Financial PhraseBank** dataset (Malo et al., 2014), pushed to **HuggingFace Hub**, and deployed via a 4-page **Streamlit** app that supports single-headline prediction, batch analysis, live RSS news scoring, and **attention-based explainability**.

🤗 **Model:** [`Tavish15100/distilbert-financial-phrasebank`](https://huggingface.co/Tavish15100/distilbert-financial-phrasebank)
🌐 **Live demo:** https://financial-news-sentiment-analysis-tavish.streamlit.app/

---

## Architecture

```
┌──────────────────────────┐
│  Financial PhraseBank    │  3,453 expert-annotated sentences
│  (75% agreement subset)  │  (HuggingFace datasets)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Fine-tune DistilBERT    │  4 epochs, AdamW, lr=2e-5, batch=16
│  (Google Colab T4 GPU)   │  ~5 min training time
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  HuggingFace Hub         │  Tavish15100/distilbert-financial-phrasebank
│  (model + tokenizer)     │
└────────────┬─────────────┘
             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Streamlit Cloud — 4 pages                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────┐ │
│  │ Single       │  │ Batch        │  │ Live News    │  │ XAI  │ │
│  │ headline     │  │ (CSV upload) │  │ (RSS feeds)  │  │      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
financial-news-sentiment/
├── src/
│   ├── config.py             # Central config (model names, hyperparams, RSS feeds)
│   ├── data.py               # Financial PhraseBank loader + train/val/test splitter
│   ├── model.py              # HF model + tokenizer loading
│   ├── train.py              # Fine-tuning pipeline (Trainer-based)
│   ├── predict.py            # SentimentPredictor batch-inference wrapper
│   ├── explainability.py     # Attention-based token attribution
│   └── news_feed.py          # RSS feed parser (Reuters, MarketWatch, Yahoo, CNBC)
├── notebooks/
│   ├── 01_exploration.ipynb         # EDA: class balance, length distribution, top words
│   ├── 02_fine_tuning.ipynb         # Colab-ready training notebook
│   └── 03_evaluation.ipynb          # Test-set metrics, confusion matrix, baseline comparison
├── streamlit_app/
│   ├── app.py                       # Landing page
│   ├── pages/
│   │   ├── 1_📝_Single_Headline.py    # Single prediction with confidence bars
│   │   ├── 2_📋_Batch_Analysis.py     # Paste/upload many headlines, aggregate stats
│   │   ├── 3_📰_Live_News.py          # Pull RSS feeds + score in real time
│   │   └── 4_🔍_Explainability.py     # Token-level attention visualization
│   └── requirements.txt             # Lightweight deps for Streamlit Cloud
├── reports/
│   └── model_card.md                # Model card with performance + intended use
├── tests/
│   └── test_predict.py              # Smoke tests for inference wrapper
├── .github/workflows/ci.yml         # Lint + tests on every push
└── requirements.txt
```

---

## What's new vs the original version

This is a **major overhaul** of the original [single-script app](https://github.com/Tavish1510/Financial-News-sentiment-analysis/tree/v0). The deployed app now uses **the actual fine-tuned model** (previously it loaded an unrelated public model), and the project is restructured into a maintainable codebase.

| | Before | After |
|---|---|---|
| **Model used in app** | Public `mrm8488` model (not fine-tuned by author) | Fine-tuned by author, pushed to HuggingFace Hub |
| **Project structure** | Single `app.py`, copied helper file | Modular `src/`, notebooks, multi-page app |
| **Training framework** | TensorFlow / Keras | PyTorch + HuggingFace Trainer |
| **App pages** | 1 (text-or-URL input) | 4 (single, batch, live news, explainability) |
| **Live news** | URL scraping (newspaper3k) | RSS feeds from 5 financial sources, cached |
| **Evaluation** | Accuracy only | Confusion matrix, per-class P/R/F1, baseline comparison |
| **Explainability** | None | Attention-based token attribution, color-coded |
| **CI** | None | GitHub Actions: ruff + pytest |
| **Deployment** | App loads wrong model | Fine-tuned model on HF Hub, fallback for resilience |

---

## Setup

### Run the Streamlit app locally

```bash
git clone https://github.com/Tavish1510/Financial-News-sentiment-analysis.git
cd Financial-News-sentiment-analysis

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

### Re-train the model (Colab, free GPU)

1. Open `notebooks/02_fine_tuning.ipynb` in Google Colab (`File → Open notebook → GitHub tab`)
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`
4. Provide your HuggingFace write token when prompted
5. Update `src/config.py` with your HF Hub repo name
6. Commit & push — the Streamlit app picks it up

Training time: ~5 min on T4.

---

## Results

See `notebooks/03_evaluation.ipynb` for the full report. Headline numbers (10% held-out test split, fill in after training):

| Metric | TF-IDF + Logistic Regression baseline | Fine-tuned DistilBERT |
|---|---|---|
| Accuracy | _baseline_ | _bert_ |
| F1 (macro) | _baseline_ | _bert_ |
| F1 (weighted) | _baseline_ | _bert_ |

The confusion matrix is saved to `reports/figures/confusion_matrix.png`.

---

## Tech stack

- **Modeling:** PyTorch, HuggingFace Transformers, HuggingFace Datasets, Trainer
- **Evaluation:** scikit-learn, evaluate, matplotlib, seaborn
- **App:** Streamlit, Plotly
- **News ingestion:** feedparser (RSS)
- **Explainability:** native transformer attention weights (Captum-compatible)
- **CI:** GitHub Actions (ruff + pytest)
- **Deployment:** HuggingFace Hub (model), Streamlit Community Cloud (app)
