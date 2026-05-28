import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import NEWS_FEEDS
from src.news_feed import fetch_all_feeds
from src.predict import SentimentPredictor

st.set_page_config(page_title="Live News", page_icon="📰", layout="wide")

st.title("📰 Live Financial News Sentiment")
st.caption("Pulls real-time headlines from major financial news RSS feeds and scores them.")


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


predictor = get_predictor()
st.caption(f"Model: `{predictor.source}`")

with st.sidebar:
    st.header("Feed settings")
    selected_feeds = st.multiselect("Active feeds", list(NEWS_FEEDS.keys()), default=list(NEWS_FEEDS.keys()))
    headlines_per_feed = st.slider("Headlines per feed", 5, 30, 15)
    if st.button("🔄 Refresh"):
        st.cache_data.clear()


@st.cache_data(ttl=300)
def fetch_and_score(selected: tuple, n: int):
    items = fetch_all_feeds(limit_per_feed=n)
    items = [it for it in items if it.source in selected]
    if not items:
        return pd.DataFrame()
    preds = predictor.predict_batch([it.title for it in items])
    return pd.DataFrame(
        [{
            "source": it.source,
            "headline": it.title,
            "sentiment": p.label,
            "confidence": p.confidence,
            "neg": p.scores.get("negative", 0),
            "neutral": p.scores.get("neutral", 0),
            "pos": p.scores.get("positive", 0),
            "url": it.url,
            "published": it.published,
        } for it, p in zip(items, preds)]
    )


if selected_feeds:
    with st.spinner("Fetching headlines and scoring..."):
        df = fetch_and_score(tuple(selected_feeds), headlines_per_feed)

    if df.empty:
        st.warning("No headlines fetched. Try increasing headlines per feed or check network.")
    else:
        # ---- KPIs ----
        counts = df["sentiment"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Headlines", len(df))
        c2.metric("🟢 Positive", int(counts.get("positive", 0)))
        c3.metric("⚪ Neutral", int(counts.get("neutral", 0)))
        c4.metric("🔴 Negative", int(counts.get("negative", 0)))

        # ---- Charts ----
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(values=counts.values, names=counts.index, hole=0.4,
                          color=counts.index,
                          color_discrete_map={"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"},
                          title="Overall sentiment distribution")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            by_source = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
            fig = px.bar(by_source, x="source", y="count", color="sentiment",
                          color_discrete_map={"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"},
                          title="Sentiment by source", barmode="stack")
            fig.update_xaxes(tickangle=20)
            st.plotly_chart(fig, use_container_width=True)

        # ---- Headlines ----
        st.subheader("Headlines")
        for _, row in df.sort_values("confidence", ascending=False).iterrows():
            emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[row["sentiment"]]
            with st.expander(f"{emoji} [{row['source']}] {row['headline']} — {row['confidence']:.1%}"):
                st.markdown(
                    f"**Sentiment scores** — "
                    f"🔴 {row['neg']:.2f}  ⚪ {row['neutral']:.2f}  🟢 {row['pos']:.2f}"
                )
                st.markdown(f"**Published:** {row['published']}")
                st.markdown(f"[Read article →]({row['url']})")
