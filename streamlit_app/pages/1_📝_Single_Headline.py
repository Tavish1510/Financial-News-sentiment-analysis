import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from src.predict import SentimentPredictor

st.set_page_config(page_title="Single Headline", page_icon="📝", layout="wide")

st.title("📝 Single Headline Analysis")


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


predictor = get_predictor()
st.caption(f"Model: `{predictor.source}`")

text = st.text_area(
    "Enter a financial headline or sentence:",
    "Tesla reports record quarterly earnings, beating analyst estimates by 12%.",
    height=100,
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            pred = predictor.predict(text)

        color_map = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
        st.subheader(f"{color_map[pred.label]} {pred.label.upper()} ({pred.confidence:.1%} confidence)")

        scores_df = [{"label": k, "probability": v} for k, v in pred.scores.items()]
        fig = px.bar(
            scores_df, x="label", y="probability", color="label",
            color_discrete_map={"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"},
            title="Class probabilities",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
