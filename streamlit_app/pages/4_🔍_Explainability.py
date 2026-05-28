import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.explainability import token_attributions
from src.predict import SentimentPredictor

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Token-level Explainability")
st.caption(
    "Shows which tokens drove the prediction. Importance is the attention from the "
    "`[CLS]` token to each input token in the final transformer layer, averaged across heads."
)


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


predictor = get_predictor()
st.caption(f"Model: `{predictor.source}`")

example_options = {
    "Bullish earnings": "Apple posted record quarterly revenue, beating Wall Street expectations.",
    "Bearish outlook": "Tesla shares plunged after the company warned of slowing demand and margin pressure.",
    "Mixed signals": "The Fed kept rates unchanged while signaling possible cuts later this year.",
    "Neutral filing": "The company filed its annual 10-K report with the SEC on Tuesday.",
}

choice = st.selectbox("Pick an example or write your own below:", ["(custom)"] + list(example_options.keys()))
default_text = example_options.get(choice, "")
text = st.text_area("Text:", default_text or "Tesla stock surged 8% after record earnings.", height=100)

if st.button("Explain", type="primary"):
    with st.spinner("Computing attention..."):
        tokens, importance, pred_label, scores = token_attributions(predictor, text)

    color_map = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
    st.subheader(f"Prediction: {color_map[pred_label]} {pred_label.upper()} ({scores[pred_label]:.1%})")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Negative", f"{scores.get('negative', 0):.1%}")
    c2.metric("⚪ Neutral", f"{scores.get('neutral', 0):.1%}")
    c3.metric("🟢 Positive", f"{scores.get('positive', 0):.1%}")

    # Filter out special tokens
    display_tokens = []
    display_importance = []
    for tok, imp in zip(tokens, importance):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        display_tokens.append(tok.replace("##", ""))
        display_importance.append(imp)

    df = pd.DataFrame({"token": display_tokens, "importance": display_importance})
    df = df.sort_values("importance", ascending=False)

    st.subheader("Token importance")
    c1, c2 = st.columns([3, 2])
    with c1:
        # Inline highlighted text
        max_imp = max(display_importance) if display_importance else 1.0
        chunks = []
        for tok, imp in zip(display_tokens, display_importance):
            opacity = min(0.95, imp / max_imp)
            color = {"negative": "231, 76, 60", "neutral": "149, 165, 166", "positive": "39, 174, 96"}[pred_label]
            chunks.append(
                f'<span style="background-color: rgba({color}, {opacity:.2f}); '
                f'padding: 2px 4px; margin: 1px; border-radius: 3px;">{tok}</span>'
            )
        st.markdown(" ".join(chunks), unsafe_allow_html=True)

    with c2:
        fig = px.bar(df.head(12), x="importance", y="token", orientation="h",
                      title="Top tokens by attention", color_discrete_sequence=["#3498DB"])
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Method"):
        st.markdown(
            """
**Attention-based attribution.** For each token we report how much the `[CLS]` token attends to it
in the final transformer layer, averaged over all attention heads. This is a fast, model-internal
signal that correlates with (but doesn't exactly equal) gradient-based attribution.

For a more rigorous attribution, swap in **Captum's Integrated Gradients** — the explainability
module exposes a stable interface so the upgrade is one-file.
"""
        )
