"""
train.py — Train and evaluate three sentiment-analysis models on IMDB reviews.

Usage:
    python train.py            # expects imdb_dataset.csv in the same directory
    python train.py --data /path/to/imdb_dataset.csv

Outputs:
    model.pkl          — best model (full pipeline: vectoriser + classifier)
    vectorizer.pkl     — TF-IDF vectoriser of the best model
    model_lr.pkl       — Logistic Regression pipeline
    model_nb.pkl       — Naive Bayes pipeline
    model_svc.pkl      — Linear SVC pipeline
    metrics.json       — evaluation metrics for all three models
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tabulate import tabulate

from preprocess import TextPreprocessor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(filename: str) -> Path:
    """Return absolute path relative to *this* script's directory."""
    return Path(__file__).resolve().parent / filename


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(data_path: str | None = None) -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = Path(data_path) if data_path else script_dir / "imdb_dataset.csv"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  SentimentIQ — Model Training Pipeline")
    print("=" * 60)
    print(f"\n📂 Loading dataset from: {data_file}")

    if not data_file.exists():
        sys.exit(f"❌ Dataset not found: {data_file}")

    df = pd.read_csv(data_file)
    print(f"   Rows loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("\n🔧 Preprocessing reviews …")
    preprocessor = TextPreprocessor()
    df["cleaned_review"] = df["review"].astype(str).apply(preprocessor.full_pipeline)

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_review"],
        df["label"],
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )
    print(f"   Train size : {len(X_train):,}")
    print(f"   Test  size : {len(X_test):,}")

    # ------------------------------------------------------------------
    # 5. Define pipelines
    # ------------------------------------------------------------------
    pipelines: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1_000, C=1.0, solver="lbfgs")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15_000, ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVC": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LinearSVC(C=1.0, max_iter=2_000)),
        ]),
    }

    # ------------------------------------------------------------------
    # 6–7. Train & evaluate
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    best_name, best_f1 = "", 0.0

    for name, pipe in pipelines.items():
        print(f"\n🚀 Training {name} …")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": cr,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    # ------------------------------------------------------------------
    # 8. Print comparison table
    # ------------------------------------------------------------------
    table_rows = []
    for name, m in results.items():
        marker = " ★" if name == best_name else ""
        table_rows.append([
            f"{name}{marker}",
            f"{m['accuracy']:.2%}",
            f"{m['precision']:.2%}",
            f"{m['recall']:.2%}",
            f"{m['f1_score']:.2%}",
        ])

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(tabulate(
        table_rows,
        headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score"],
        tablefmt="fancy_grid",
    ))
    print(f"\n🏆 Best model: {best_name} (F1 = {best_f1:.4f})")

    # ------------------------------------------------------------------
    # 9. Save artefacts
    # ------------------------------------------------------------------
    best_pipe = pipelines[best_name]
    joblib.dump(best_pipe, script_dir / "model.pkl")
    joblib.dump(best_pipe.named_steps["tfidf"], script_dir / "vectorizer.pkl")
    print(f"\n💾 Saved model.pkl  (best: {best_name})")
    print("💾 Saved vectorizer.pkl")

    # Save each model individually
    short = {"Logistic Regression": "lr", "Naive Bayes": "nb", "Linear SVC": "svc"}
    for name, pipe in pipelines.items():
        fname = f"model_{short[name]}.pkl"
        joblib.dump(pipe, script_dir / fname)
        print(f"💾 Saved {fname}")

    # ------------------------------------------------------------------
    # 10. Save metrics
    # ------------------------------------------------------------------
    metrics_payload = {
        "best_model": best_name,
        "dataset_size": len(df),
        "positive_count": int(df["label"].sum()),
        "negative_count": int((df["label"] == 0).sum()),
        "avg_review_length": round(df["review"].astype(str).str.len().mean(), 1),
        "review_lengths": df["review"].astype(str).str.len().tolist(),
        "models": results,
    }
    with open(script_dir / "metrics.json", "w") as fh:
        json.dump(metrics_payload, fh, indent=2)
    print("💾 Saved metrics.json")
    print("\n✅ Training complete!\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentimentIQ models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to imdb_dataset.csv")
    args = parser.parse_args()
    main(data_path=args.data)
