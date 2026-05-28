import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import SentimentPredictor

st.set_page_config(page_title="Batch Analysis", page_icon="📋", layout="wide")

st.title("📋 Batch Headline Analysis")


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


predictor = get_predictor()
st.caption(f"Model: `{predictor.source}`")

DEFAULT_BATCH = """Tesla stock surges 8% on record Q3 deliveries
Federal Reserve hints at additional rate hikes amid inflation worries
Microsoft Cloud revenue beats expectations, lifting quarterly profit
Layoffs across the tech industry continue as Meta cuts another 5,000 jobs
Apple maintains steady iPhone sales despite weakening global demand
Oil prices tumble as OPEC+ fails to reach output agreement"""

text_input = st.text_area(
    "Paste headlines (one per line):",
    DEFAULT_BATCH,
    height=200,
)

uploaded = st.file_uploader("...or upload a CSV/TXT file", type=["csv", "txt"])

headlines: list[str] = []
if uploaded:
    if uploaded.name.endswith(".csv"):
        df_in = pd.read_csv(uploaded)
        col = st.selectbox("Which column has the text?", df_in.columns.tolist())
        headlines = df_in[col].dropna().astype(str).tolist()
    else:
        headlines = [line.strip() for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]
elif text_input.strip():
    headlines = [line.strip() for line in text_input.splitlines() if line.strip()]

if st.button("Analyze batch", type="primary") and headlines:
    with st.spinner(f"Scoring {len(headlines):,} headlines..."):
        preds = predictor.predict_batch(headlines)

    df = pd.DataFrame(
        [{
            "headline": p.text,
            "sentiment": p.label,
            "confidence": p.confidence,
            "neg": p.scores.get("negative", 0),
            "neutral": p.scores.get("neutral", 0),
            "pos": p.scores.get("positive", 0),
        } for p in preds]
    )

    # ----- aggregate metrics -----
    counts = df["sentiment"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("🟢 Positive", int(counts.get("positive", 0)))
    c3.metric("⚪ Neutral", int(counts.get("neutral", 0)))
    c4.metric("🔴 Negative", int(counts.get("negative", 0)))

    # ----- distribution pie -----
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(values=counts.values, names=counts.index, hole=0.4,
                      color=counts.index,
                      color_discrete_map={"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"},
                      title="Sentiment distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        avg = df[["neg", "neutral", "pos"]].mean()
        avg_df = pd.DataFrame({"label": ["negative", "neutral", "positive"], "avg_probability": avg.values})
        fig = px.bar(avg_df, x="label", y="avg_probability", color="label",
                      color_discrete_map={"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"},
                      title="Average class probability across batch")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # ----- per-headline table -----
    st.subheader("Per-headline results")
    display = df[["headline", "sentiment", "confidence"]].copy()
    display["confidence"] = display["confidence"].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv, "sentiment_results.csv", "text/csv")
