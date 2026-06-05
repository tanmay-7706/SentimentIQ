"""
app.py — SentimentIQ: a professional Streamlit dashboard for movie-review
sentiment analysis.

Run with:
    streamlit run app.py
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import TextPreprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pkl"
VECTORIZER_PATH = SCRIPT_DIR / "vectorizer.pkl"
METRICS_PATH = SCRIPT_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- result cards ---- */
.pos-card {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    border-left: 6px solid #28a745;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
    box-shadow: 0 2px 12px rgba(40,167,69,.15);
}
.neg-card {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    border-left: 6px solid #dc3545;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
    box-shadow: 0 2px 12px rgba(220,53,69,.15);
}
.pos-card h2, .neg-card h2 { margin: 0 0 4px 0; }
.pos-card p,  .neg-card p  { margin: 0; font-size: 1.05rem; }

/* ---- metric cards ---- */
.metric-card {
    background: #f7f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.metric-card .label { font-size: 0.82rem; color: #6a737d; margin-bottom: 4px; }
.metric-card .value { font-size: 1.55rem; font-weight: 700; color: #24292e; }

/* ---- word chips ---- */
.chip-pos {
    display: inline-block;
    background: #d4edda; color: #155724;
    border-radius: 20px; padding: 4px 14px;
    margin: 3px 4px; font-size: 0.88rem; font-weight: 500;
}
.chip-neg {
    display: inline-block;
    background: #f8d7da; color: #721c24;
    border-radius: 20px; padding: 4px 14px;
    margin: 3px 4px; font-size: 0.88rem; font-weight: 500;
}

/* ---- header ---- */
.app-header {
    text-align: center; padding: 10px 0 4px 0;
}
.app-header h1 { font-size: 2.2rem; margin-bottom: 2px; }
.app-header p  { color: #6a737d; font-size: 1.05rem; }

/* ---- hide default Streamlit menu hamburger ---- */
/* (optional — remove if you want the menu) */
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the best trained pipeline (vectoriser + classifier)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading vectorizer …")
def load_vectorizer():
    """Load the standalone TF-IDF vectoriser (for feature inspection)."""
    if not VECTORIZER_PATH.exists():
        return None
    return joblib.load(VECTORIZER_PATH)


@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()


def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_sentiment(model, text: str):
    """Return (label_str, positive_prob, negative_prob)."""
    preprocessor = load_preprocessor()
    cleaned = preprocessor.full_pipeline(text)
    pred = model.predict([cleaned])[0]

    clf = model.named_steps["clf"]
    # LinearSVC → decision_function; others → predict_proba
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(
            model.named_steps["tfidf"].transform([cleaned])
        )[0]
        neg_prob, pos_prob = proba[0], proba[1]
    else:
        score = clf.decision_function(
            model.named_steps["tfidf"].transform([cleaned])
        )[0]
        pos_prob = _sigmoid(score)
        neg_prob = 1.0 - pos_prob

    label = "Positive" if pred == 1 else "Negative"
    return label, float(pos_prob), float(neg_prob)


def get_top_keywords(model, text: str, top_n: int = 10) -> list[str]:
    """Return the *top_n* words from *text* that rank highest in TF-IDF."""
    preprocessor = load_preprocessor()
    cleaned = preprocessor.full_pipeline(text)
    vectorizer = model.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()

    vec = vectorizer.transform([cleaned])
    scores = vec.toarray().flatten()

    # Keep only unigrams that actually appear in the cleaned text
    words_in_text = set(cleaned.split())
    candidates = [
        (feature_names[i], scores[i])
        for i in scores.argsort()[::-1]
        if scores[i] > 0 and feature_names[i] in words_in_text
    ]
    return [w for w, _ in candidates[:top_n]]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎬 SentimentIQ")
    st.caption("AI-powered movie-review sentiment analysis")
    st.divider()

    metrics = load_metrics()
    if metrics:
        best = metrics["best_model"]
        best_metrics = metrics["models"][best]

        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Best Model Accuracy</div>'
            f'<div class="value">{best_metrics["accuracy"]:.2%}</div>'
            '</div>', unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">F1 Score</div>'
            f'<div class="value">{best_metrics["f1_score"]:.2%}</div>'
            '</div>', unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Total Reviews Trained On</div>'
            f'<div class="value">{metrics["dataset_size"]:,}</div>'
            '</div>', unsafe_allow_html=True,
        )
    st.markdown("")

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Preprocessing** — HTML, URLs, stopwords and noise are removed; "
            "words are stemmed.\n"
            "2. **Vectorisation** — TF-IDF converts text into a numerical feature "
            "matrix (up to 15 000 n-grams).\n"
            "3. **Classification** — A trained ML model predicts "
            "*positive* or *negative* sentiment.\n"
            "4. **Confidence** — Probability scores show how certain "
            "the model is about its prediction."
        )

    st.divider()
    st.caption("Built with ❤️ using Streamlit & scikit-learn")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    '<h1>🎬 SentimentIQ</h1>'
    '<p>Instantly analyze the sentiment of any movie review</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (or show training prompt)
# ---------------------------------------------------------------------------
model = load_model()
if model is None:
    st.warning("⚠️ No trained model found. Please train the model first.")
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training models — this may take a few minutes …"):
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "train.py")],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                st.success("✅ Training complete! Reloading …")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Training failed:")
                st.code(result.stderr or result.stdout)
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Review",
    "📊 Batch Analysis",
    "📈 Model Insights",
])

# ===== TAB 1 — Single review =============================================
with tab1:
    review_text = st.text_area(
        "Enter a movie review",
        height=150,
        placeholder="Write a movie review and let SentimentIQ predict its sentiment …",
        max_chars=2000,
        key="review_input",
    )
    st.caption(f"Characters: {len(review_text)} / 2 000")

    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing …"):
                label, pos_prob, neg_prob = predict_sentiment(model, review_text)
                confidence = max(pos_prob, neg_prob)

                # Styled card
                emoji = "😊" if label == "Positive" else "😞"
                css_class = "pos-card" if label == "Positive" else "neg-card"
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<h2>{emoji} {label}</h2>'
                    f'<p>Confidence: <strong>{confidence:.1%}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Plotly bar
                fig_bar = go.Figure(go.Bar(
                    x=[pos_prob, neg_prob],
                    y=["Positive", "Negative"],
                    orientation="h",
                    marker_color=["#28a745", "#dc3545"],
                    text=[f"{pos_prob:.1%}", f"{neg_prob:.1%}"],
                    textposition="auto",
                ))
                fig_bar.update_layout(
                    title="Sentiment Probability",
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1]),
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Key words
                keywords = get_top_keywords(model, review_text)
                if keywords:
                    st.subheader("🔑 Key Words That Influenced Prediction")
                    chip_class = "chip-pos" if label == "Positive" else "chip-neg"
                    chips_html = " ".join(
                        f'<span class="{chip_class}">{w}</span>' for w in keywords
                    )
                    st.markdown(chips_html, unsafe_allow_html=True)


# ===== TAB 2 — Batch analysis =============================================
with tab2:
    uploaded = st.file_uploader(
        "Upload a CSV with a 'review' column", type=["csv"],
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            st.stop()

        if "review" not in batch_df.columns:
            st.error("The CSV must contain a column named **review**.")
        else:
            st.subheader("📋 Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Analyze All", type="primary", use_container_width=True):
                progress = st.progress(0, text="Processing …")
                sentiments, confidences = [], []

                for idx, row in enumerate(batch_df["review"].astype(str)):
                    label, pos_p, neg_p = predict_sentiment(model, row)
                    sentiments.append(label)
                    confidences.append(round(max(pos_p, neg_p) * 100, 1))
                    progress.progress(
                        (idx + 1) / len(batch_df),
                        text=f"Processing review {idx + 1}/{len(batch_df)} …",
                    )

                batch_df["review_preview"] = batch_df["review"].astype(str).str[:50] + "…"
                batch_df["sentiment"] = sentiments
                batch_df["confidence (%)"] = confidences

                progress.empty()
                st.success("✅ Batch analysis complete!")

                # Color-code via Styler
                def _highlight(row):
                    colour = "background-color: #d4edda" if row["sentiment"] == "Positive" \
                        else "background-color: #f8d7da"
                    return [colour] * len(row)

                display_cols = ["review_preview", "sentiment", "confidence (%)"]
                styled = batch_df[display_cols].style.apply(_highlight, axis=1)
                st.dataframe(styled, use_container_width=True)

                # Summary
                n_pos = sentiments.count("Positive")
                n_neg = sentiments.count("Negative")
                avg_conf = np.mean(confidences)

                c1, c2, c3 = st.columns(3)
                c1.metric("👍 Positive", n_pos)
                c2.metric("👎 Negative", n_neg)
                c3.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")

                # Pie chart
                fig_pie = px.pie(
                    names=["Positive", "Negative"],
                    values=[n_pos, n_neg],
                    color_discrete_sequence=["#28a745", "#dc3545"],
                    title="Sentiment Distribution",
                    hole=0.4,
                )
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                    template="plotly_white",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Download
                csv_bytes = batch_df[display_cols].to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name="sentimentiq_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ===== TAB 3 — Model Insights =============================================
with tab3:
    metrics = load_metrics()
    if metrics is None:
        st.info(
            "📭 No metrics found. Run **`python train.py`** first to train "
            "the models and generate performance metrics."
        )
    else:
        # --- Comparison table ---
        st.subheader("📊 Model Performance Comparison")

        model_names = list(metrics["models"].keys())
        comp_data = []
        for name in model_names:
            m = metrics["models"][name]
            comp_data.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
            })
        comp_df = pd.DataFrame(comp_data)

        # Highlight best in each column
        def _best_highlight(s):
            is_best = s == s.max()
            return ["font-weight: bold; color: #28a745" if v else "" for v in is_best]

        numeric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        styled_comp = (
            comp_df.style
            .apply(_best_highlight, subset=numeric_cols)
            .format({c: "{:.2%}" for c in numeric_cols})
        )
        st.dataframe(styled_comp, use_container_width=True, hide_index=True)

        # --- Grouped bar chart ---
        melted = comp_df.melt(id_vars="Model", value_vars=numeric_cols,
                              var_name="Metric", value_name="Score")
        fig_group = px.bar(
            melted, x="Metric", y="Score", color="Model",
            barmode="group",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
            title="Metric Comparison Across Models",
        )
        fig_group.update_layout(
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_group, use_container_width=True)

        # --- Confusion matrix heatmap (best model) ---
        st.subheader(f"🔥 Confusion Matrix — {metrics['best_model']}")
        cm = np.array(metrics["models"][metrics["best_model"]]["confusion_matrix"])
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{metrics['best_model']} — Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)

        # --- Training data insights ---
        st.subheader("📚 Training Data Insights")
        d1, d2, d3 = st.columns(3)
        d1.metric("Dataset Size", f"{metrics['dataset_size']:,}")
        d2.metric("Positive Reviews", f"{metrics['positive_count']:,}")
        d3.metric("Negative Reviews", f"{metrics['negative_count']:,}")

        st.metric("Average Review Length", f"{metrics['avg_review_length']:.0f} chars")

        # Review length distribution
        if "review_lengths" in metrics:
            fig_hist = px.histogram(
                x=metrics["review_lengths"],
                nbins=60,
                title="Review Length Distribution",
                labels={"x": "Character Length", "y": "Count"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_hist.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
