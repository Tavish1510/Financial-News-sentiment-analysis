"""Multi-page Streamlit app for the Financial News Sentiment project.

Pages are auto-discovered from the `pages/` directory by Streamlit.
This file is the landing page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Financial News Sentiment",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial News Sentiment Analysis")
st.caption("DistilBERT fine-tuned on Financial PhraseBank, deployed via HuggingFace Hub + Streamlit")

st.markdown(
    """
## What this is

A fine-tuned **DistilBERT** model that classifies financial sentences as **negative**, **neutral**, or **positive**.

Use the sidebar to navigate to:

| Page | What it does |
|---|---|
| **📝 Single Headline** | Type or paste one headline → get sentiment + confidence |
| **📋 Batch Analysis** | Paste many headlines at once → aggregate sentiment + per-headline breakdown |
| **📰 Live News** | Pull live headlines from Reuters / MarketWatch / Yahoo Finance / CNBC and score them |
| **🔍 Explainability** | See which tokens drove the prediction via attention weights |

---

## Architecture

```
Financial PhraseBank ──► Fine-tune DistilBERT ──► Push to HuggingFace Hub
                                                          │
                                                          ▼
                                            Streamlit Cloud loads model
                                                          │
                       ┌──────────────────────────────────┼──────────────────────┐
                       ▼                                  ▼                      ▼
              Single prediction              Batch analysis              Live RSS feeds
              (any headline)                 (paste many)                 + scoring
                       │                                  │                      │
                       └──────────────────► Attention visualization ◄────────────┘
                                              (explainability)
```

## Model

- **Base model:** `distilbert-base-uncased` (66M params)
- **Training data:** Financial PhraseBank (75% agreement subset, ~3,400 sentences)
- **Fine-tuning:** 4 epochs, batch size 16, learning rate 2e-5, AdamW, weight decay 0.01
- **Evaluation:** Held-out 10% test split

See `notebooks/03_evaluation.ipynb` and `reports/model_card.md` for full performance details.
"""
)

st.sidebar.success("Pick a page from the sidebar above ↑")
