"""Central configuration for the Financial News Sentiment project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"

# HuggingFace Hub repo where the fine-tuned weights are pushed after training.
# Update this to your own HF Hub user after running the training notebook.
HF_HUB_REPO = "Tavish1510/distilbert-financial-phrasebank"

# Fall back to a public pre-trained financial sentiment model if HF_HUB_REPO
# is unavailable (so the Streamlit demo still works for anyone cloning).
FALLBACK_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# ---------------------------------------------------------------------------
# Labels — Financial PhraseBank uses 3 classes
# ---------------------------------------------------------------------------
LABEL_LIST = ["negative", "neutral", "positive"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# News feeds for the live-news Streamlit page
# ---------------------------------------------------------------------------
NEWS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch Top Stories": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Investing.com Stock News": "https://www.investing.com/rss/news_25.rss",
}
