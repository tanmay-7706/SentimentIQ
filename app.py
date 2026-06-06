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
# Google Fonts
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Playful Geometric — Full CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #FFFDF5 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* HEADINGS */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    color: #1E293B !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #FFFDF5 !important;
    border-right: 2px solid #1E293B !important;
}

/* ALL BUTTONS */
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    background-color: #8B5CF6 !important;
    color: #FFFFFF !important;
    border: 2px solid #1E293B !important;
    border-radius: 9999px !important;
    box-shadow: 4px 4px 0px #1E293B !important;
    padding: 0.5rem 2rem !important;
    transition: all 200ms cubic-bezier(0.34,1.56,0.64,1) !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0px #1E293B !important;
    background-color: #7C3AED !important;
}
.stButton > button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0px #1E293B !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    background-color: #34D399 !important;
    color: #1E293B !important;
    border: 2px solid #1E293B !important;
    border-radius: 9999px !important;
    box-shadow: 4px 4px 0px #1E293B !important;
    padding: 0.5rem 2rem !important;
    transition: all 200ms cubic-bezier(0.34,1.56,0.64,1) !important;
    letter-spacing: 0.03em !important;
}
.stDownloadButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0px #1E293B !important;
    background-color: #2DD4BF !important;
}

/* TEXT AREA & INPUTS */
.stTextArea textarea, .stTextInput input {
    border: 2px solid #1E293B !important;
    border-radius: 16px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 4px 4px 0px #8B5CF6 !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    background-color: #F1F5F9 !important;
    border: 2px solid #1E293B !important;
    border-radius: 9999px !important;
    padding: 0.3rem 1.4rem !important;
    box-shadow: 3px 3px 0px #1E293B !important;
    transition: all 200ms cubic-bezier(0.34,1.56,0.64,1) !important;
}
.stTabs [aria-selected="true"] {
    background-color: #8B5CF6 !important;
    color: #FFFFFF !important;
    box-shadow: 3px 3px 0px #1E293B !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 2px solid #1E293B !important;
    border-radius: 16px !important;
    box-shadow: 4px 4px 0px #1E293B !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    color: #8B5CF6 !important;
}

/* EXPANDERS */
details > summary {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: 2px solid #1E293B !important;
    border-radius: 12px !important;
    background: #F1F5F9 !important;
    padding: 0.6rem 1rem !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 2px dashed #8B5CF6 !important;
    border-radius: 16px !important;
    background-color: #F1F5F9 !important;
    padding: 1rem !important;
}

/* PROGRESS BAR */
[data-testid="stProgressBar"] > div {
    background-color: #8B5CF6 !important;
    border-radius: 9999px !important;
}

/* SELECT BOX */
.stSelectbox [data-baseweb="select"] {
    border: 2px solid #1E293B !important;
    border-radius: 12px !important;
    background: white !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 2px solid #1E293B !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ALERTS/INFO/SUCCESS/WARNING */
.stAlert {
    border-radius: 16px !important;
    border: 2px solid #1E293B !important;
    box-shadow: 3px 3px 0px #1E293B !important;
}

/* DIVIDER */
hr {
    border: none !important;
    height: 2px !important;
    background: repeating-linear-gradient(
        90deg,
        #8B5CF6 0px, #8B5CF6 20px,
        #F472B6 20px, #F472B6 40px,
        #FBBF24 40px, #FBBF24 60px,
        #34D399 60px, #34D399 80px
    ) !important;
    border-radius: 9999px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Playful Geometric — UI helper functions
# ---------------------------------------------------------------------------

def render_header():
    """Render the main page header with decorative geometric shapes."""
    st.markdown("""
    <div style="
        background:#FFFDF5; padding:1.75rem 2rem 1.25rem;
        border-bottom:2px solid #1E293B; margin-bottom:1.5rem;
        position:relative; overflow:hidden;
    ">
        <div style="position:absolute;top:-15px;right:100px;
            width:90px;height:90px;background:#FBBF24;
            border-radius:50%;border:2px solid #1E293B;opacity:0.75;"></div>
        <div style="position:absolute;top:15px;right:38px;
            width:44px;height:44px;background:#F472B6;
            border-radius:50%;border:2px solid #1E293B;opacity:0.85;"></div>
        <div style="position:absolute;bottom:8px;left:240px;
            width:55px;height:55px;background:#34D399;
            border-radius:50%;border:2px solid #1E293B;opacity:0.65;"></div>
        <div style="position:absolute;bottom:8px;left:120px;
            width:20px;height:20px;background:#8B5CF6;
            border-radius:4px;transform:rotate(45deg);
            border:2px solid #1E293B;opacity:0.6;"></div>
        <div style="position:absolute;bottom:12px;left:60px;
            width:14px;height:14px;background:#8B5CF6;
            transform:rotate(45deg);border:2px solid #1E293B;
            opacity:0.7;"></div>
        <h1 style="
            font-family:'Outfit',sans-serif;font-size:2.5rem;
            font-weight:800;color:#1E293B;margin:0;
            position:relative;z-index:1;
        ">🎬 SentimentIQ</h1>
        <p style="
            font-family:'Plus Jakarta Sans',sans-serif;
            font-size:1rem;color:#64748B;margin:0.3rem 0 0;
            position:relative;z-index:1;
        ">Movie Review Sentiment Analyzer — Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_header():
    """Render the sidebar branded header card."""
    st.sidebar.markdown("""
    <div style="
        background:#8B5CF6;padding:1.25rem 1.25rem 1rem;padding-left:0.5rem;
        border:2px solid #1E293B;border-radius:20px;
        box-shadow:4px 4px 0px #1E293B;margin-bottom:1rem;
        position:relative;overflow:hidden;
    ">
        <div style="position:absolute;top:-18px;right:-12px;
            width:64px;height:64px;background:#FBBF24;
            border-radius:50%;border:2px solid #1E293B;opacity:0.8;"></div>
        <h2 style="
            font-family:'Outfit',sans-serif;font-size:1.4rem;
            font-weight:800;color:white;margin:0;position:relative;z-index:1;
        ">🎬 SentimentIQ</h2>
        <p style="
            font-family:'Plus Jakarta Sans',sans-serif;
            font-size:0.8rem;color:#EDE9FE;
            margin:0.2rem 0 0;position:relative;z-index:1;
        ">ML-Powered Movie Sentiment</p>
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_result(sentiment: str, confidence: float):
    """Render a styled sentiment result card."""
    if sentiment == "positive":
        bg = "#ECFDF5"; circle = "#34D399"; emoji = "😊"
        label = "POSITIVE"; text_col = "#065F46"; sub_col = "#047857"
        badge_bg = "#34D399"
    else:
        bg = "#FFF1F2"; circle = "#F472B6"; emoji = "😞"
        label = "NEGATIVE"; text_col = "#9F1239"; sub_col = "#BE123C"
        badge_bg = "#F472B6"

    st.markdown(f"""
    <div style="
        background:{bg};border:2px solid #1E293B;
        border-radius:20px;box-shadow:6px 6px 0px #1E293B;
        padding:1.5rem 1.75rem;margin:1rem 0;
        display:flex;align-items:center;gap:1.25rem;
    ">
        <div style="
            width:60px;height:60px;background:{circle};
            border-radius:50%;border:2px solid #1E293B;
            display:flex;align-items:center;justify-content:center;
            font-size:1.75rem;flex-shrink:0;
            box-shadow:3px 3px 0px #1E293B;
        ">{emoji}</div>
        <div style="flex:1;">
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:1.6rem;font-weight:800;color:{text_col};
            ">{label}</div>
            <div style="
                font-family:'Plus Jakarta Sans',sans-serif;
                color:{sub_col};font-size:0.95rem;
            ">Model is {confidence:.1f}% confident in this prediction</div>
        </div>
        <div style="
            background:{badge_bg};color:#1E293B;
            font-family:'Outfit',sans-serif;font-weight:800;
            padding:0.5rem 1.25rem;border-radius:9999px;
            border:2px solid #1E293B;box-shadow:3px 3px 0px #1E293B;
            font-size:1.1rem;flex-shrink:0;
        ">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_word_chips(words: list):
    """Render colored keyword chips in Playful Geometric style."""
    colors = [
        ("#8B5CF6", "white"), ("#F472B6", "white"),
        ("#FBBF24", "#1E293B"), ("#34D399", "#1E293B")
    ]
    chips_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:0.75rem 0;">'
    for i, word in enumerate(words):
        bg, fg = colors[i % 4]
        chips_html += f"""
        <span style="
            background:{bg};color:{fg};
            border:2px solid #1E293B;border-radius:9999px;
            padding:0.3rem 1rem;
            font-family:'Outfit',sans-serif;font-weight:700;
            font-size:0.875rem;box-shadow:2px 2px 0px #1E293B;
        ">{word}</span>"""
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_sidebar_metric(label: str, value: str, shadow_color: str = "#FBBF24"):
    """Render a styled metric card in the sidebar."""
    st.sidebar.markdown(f"""
    <div style="
        background:white;border:2px solid #1E293B;
        border-radius:14px;box-shadow:3px 3px 0px {shadow_color};
        padding:0.875rem 1.1rem;padding-left:0.5rem;margin-bottom:0.75rem;
        overflow:hidden;
    ">
        <div style="
            font-family:'Plus Jakarta Sans',sans-serif;
            font-size:0.65rem;color:#64748B;
            text-transform:uppercase;letter-spacing:0.06em;
        ">{label}</div>
        <div style="
            font-family:'Outfit',sans-serif;
            font-size:1.4rem;font-weight:800;color:#8B5CF6;
        ">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def apply_pg_plotly_theme(fig):
    """Apply Playful Geometric theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor='#FFFDF5',
        paper_bgcolor='#FFFDF5',
        font=dict(family='Plus Jakarta Sans', color='#1E293B', size=12),
        title_font=dict(family='Outfit', size=16, color='#1E293B'),
        colorway=['#8B5CF6', '#F472B6', '#FBBF24', '#34D399', '#60A5FA'],
        legend=dict(
            bgcolor='#FFFFFF', bordercolor='#1E293B', borderwidth=2
        ),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig.update_xaxes(
        gridcolor='#E2E8F0', linecolor='#1E293B',
        tickfont=dict(family='Plus Jakarta Sans', color='#1E293B')
    )
    fig.update_yaxes(
        gridcolor='#E2E8F0', linecolor='#1E293B',
        tickfont=dict(family='Plus Jakarta Sans', color='#1E293B')
    )
    return fig


def pg_divider():
    """Render a colorful Playful Geometric section divider."""
    st.markdown("""
    <div style="
        height:3px;margin:1.5rem 0;border-radius:9999px;
        background:repeating-linear-gradient(
            90deg,
            #8B5CF6 0px,#8B5CF6 24px,
            #F472B6 24px,#F472B6 48px,
            #FBBF24 48px,#FBBF24 72px,
            #34D399 72px,#34D399 96px
        );
    "></div>""", unsafe_allow_html=True)


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
    render_sidebar_header()
    pg_divider()

    metrics = load_metrics()
    if metrics:
        best = metrics["best_model"]
        best_metrics = metrics["models"][best]

        render_sidebar_metric(
            "Best Model Accuracy",
            f"{best_metrics['accuracy']:.2%}",
            shadow_color="#FBBF24",
        )
        render_sidebar_metric(
            "F1 Score",
            f"{best_metrics['f1_score']:.2%}",
            shadow_color="#F472B6",
        )
        render_sidebar_metric(
            "Total Reviews Trained On",
            f"{metrics['dataset_size']:,}",
            shadow_color="#34D399",
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

    pg_divider()
    st.caption("Built with ❤️ using Streamlit & scikit-learn")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_header()

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
    st.markdown(f'<div style="color: #64748B; font-family: Plus Jakarta Sans; font-size: 0.8rem; text-align: right; margin-top: -0.5rem; margin-bottom: 0.5rem;">Characters: {len(review_text)} / 2000</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    if st.button("Analyze Sentiment →", use_container_width=True, type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing …"):
                label, pos_prob, neg_prob = predict_sentiment(model, review_text)
                confidence = max(pos_prob, neg_prob)

                # Styled sentiment result card
                render_sentiment_result(
                    "positive" if label == "Positive" else "negative",
                    confidence * 100,
                )

                pg_divider()

                # Plotly bar
                fig_bar = go.Figure(go.Bar(
                    x=[pos_prob, neg_prob],
                    y=["Positive", "Negative"],
                    orientation="h",
                    marker_color=["#34D399", "#F472B6"],
                    text=[f"{pos_prob:.1%}", f"{neg_prob:.1%}"],
                    textposition="auto",
                    textfont=dict(family="Outfit", size=14, color="#1E293B"),
                ))
                fig_bar.update_layout(
                    title="Sentiment Probability",
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1]),
                    height=220,
                )
                apply_pg_plotly_theme(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Key words
                keywords = get_top_keywords(model, review_text)
                if keywords:
                    st.subheader("🔑 Key Words That Influenced Prediction")
                    render_word_chips(keywords)


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

                pg_divider()

                # Color-code via Styler
                def _highlight(row):
                    colour = "background-color: #ECFDF5" if row["sentiment"] == "Positive" \
                        else "background-color: #FFF1F2"
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

                pg_divider()

                # Pie chart
                fig_pie = px.pie(
                    names=["Positive", "Negative"],
                    values=[n_pos, n_neg],
                    color_discrete_sequence=["#34D399", "#F472B6"],
                    title="Sentiment Distribution",
                    hole=0.4,
                )
                apply_pg_plotly_theme(fig_pie)
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
        # --- Styled comparison header ---
        st.markdown("""
        <div style="
            background:#8B5CF6;color:white;
            font-family:'Outfit',sans-serif;font-weight:800;
            font-size:1.1rem;padding:0.75rem 1.25rem;
            border:2px solid #1E293B;border-radius:16px 16px 0 0;
            box-shadow:4px 4px 0px #1E293B;
        ">📈 Model Performance Comparison</div>
        """, unsafe_allow_html=True)

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
            return ["font-weight: bold; color: #8B5CF6" if v else "" for v in is_best]

        numeric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        styled_comp = (
            comp_df.style
            .apply(_best_highlight, subset=numeric_cols)
            .format({c: "{:.2%}" for c in numeric_cols})
        )
        st.dataframe(styled_comp, use_container_width=True, hide_index=True)

        pg_divider()

        # --- Grouped bar chart ---
        melted = comp_df.melt(id_vars="Model", value_vars=numeric_cols,
                              var_name="Metric", value_name="Score")
        fig_group = px.bar(
            melted, x="Metric", y="Score", color="Model",
            barmode="group",
            title="Metric Comparison Across Models",
        )
        fig_group.update_layout(
            yaxis=dict(range=[0, 1], tickformat=".0%"),
        )
        apply_pg_plotly_theme(fig_group)
        st.plotly_chart(fig_group, use_container_width=True)

        pg_divider()

        # --- Confusion matrix heatmap (best model) ---
        st.subheader(f"🔥 Confusion Matrix — {metrics['best_model']}")
        cm = np.array(metrics["models"][metrics["best_model"]]["confusion_matrix"])
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap=sns.color_palette(["#F1F5F9", "#C4B5FD", "#8B5CF6", "#6D28D9"]),
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
            linewidths=2,
            linecolor="#1E293B",
        )
        ax.set_xlabel("Predicted", fontfamily="sans-serif", fontsize=11)
        ax.set_ylabel("Actual", fontfamily="sans-serif", fontsize=11)
        ax.set_title(f"{metrics['best_model']} — Confusion Matrix",
                     fontfamily="sans-serif", fontsize=13, fontweight="bold")
        fig_cm.patch.set_facecolor("#FFFDF5")
        ax.set_facecolor("#FFFDF5")
        plt.tight_layout()
        st.pyplot(fig_cm)

        pg_divider()

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
            )
            apply_pg_plotly_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)
