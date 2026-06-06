# 🎬 SentimentIQ — Movie Review Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**🚀 Live Deployment:** [https://tanmay-sentimentiq.streamlit.app/](https://tanmay-sentimentiq.streamlit.app/)

## Overview

**SentimentIQ** is an end-to-end machine-learning web application that classifies
movie reviews as *positive* or *negative*. It features a stunning "Playful Geometric" 
design system, custom CSS styling, and robust NLP models trained on the IMDB 50K dataset.

## Features

- 🔍 **Single Review Analysis** — type or paste any review and get an instant sentiment prediction with confidence score
- 📊 **Batch CSV Analysis** — upload a CSV of reviews and get bulk predictions with downloadable results
- 📈 **Model Insights Dashboard** — compare Logistic Regression, Naive Bayes, Linear SVC, and a powerful **Voting Ensemble** side-by-side
- 🧹 **Advanced NLP Pipeline** — HTML stripping, URL removal, smart negation handling, stopword filtering, and Porter stemming
- 🎯 **Key Word Extraction** — see which words drove the model's decision
- 🎨 **Playful Geometric UI** — stunning Memphis Group-inspired aesthetics with vibrant colors, custom fonts (Outfit/Plus Jakarta Sans), and dynamic shapes
- ⚡ **Cross-Validation** — robust 5-fold CV evaluation for the top-performing ensemble

## Tech Stack

| Component        | Technology                        |
| ---------------- | --------------------------------- |
| Frontend         | Streamlit 1.32                    |
| ML Framework     | scikit-learn 1.4                  |
| NLP              | NLTK 3.8, BeautifulSoup4         |
| Data Processing  | pandas, NumPy                     |
| Visualisation    | Plotly, Matplotlib, Seaborn       |
| Serialisation    | joblib                            |
| Language         | Python 3.10+                      |

## ML Model Performance

*Results on the IMDB 50K test split (20 %, stratified)*

| Model               | TF-IDF Config | Accuracy | F1 Score |
| -------------------- | ------------- | -------- | -------- |
| Logistic Regression  | 20k, unigram-trigram | ~89 %    | ~89 %    |
| Multinomial Naive Bayes | 20k, unigram-trigram | ~87 % | ~87 %    |
| Linear SVC (Calibrated) | 20k, unigram-trigram | ~90 %    | ~90 %    |
| **Voting Ensemble** | **Soft Voting** | **~90 %** | **~90 %** |

> Exact numbers depend on the dataset and preprocessing. Run `python train.py` to get your own results.

## Project Structure

```
sentiment_analyzer/
├── app.py                # Streamlit web application
├── train.py              # Model training & evaluation script
├── preprocess.py         # Text preprocessing pipeline
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .streamlit/
│   └── config.toml       # Streamlit theme config
├── imdb_dataset.csv      # IMDB 50K dataset (you provide this)
├── model.pkl             # Best model (generated after training)
├── vectorizer.pkl        # TF-IDF vectorizer (generated)
├── model_lr.pkl          # Logistic Regression pipeline (generated)
├── model_nb.pkl          # Naive Bayes pipeline (generated)
├── model_svc.pkl         # Linear SVC pipeline (generated)
└── metrics.json          # Evaluation metrics (generated)
```

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/sentimentiq.git
   cd sentimentiq/sentiment_analyzer
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**

   Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   and place `imdb_dataset.csv` (with columns `review` and `sentiment`) in the
   `sentiment_analyzer/` directory.

5. **Train the models**

   ```bash
   python train.py
   ```

6. **Launch the app**

   ```bash
   streamlit run app.py
   ```

   The app will open at [http://localhost:8501](http://localhost:8501).

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `requirements.txt` is at the repo root
   or inside the `sentiment_analyzer/` directory).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repo, branch, and set the main file path to
   `sentiment_analyzer/app.py`.
4. Pre-train your model locally and commit `model.pkl`, `vectorizer.pkl`, and
   `metrics.json` to the repo (or use Git LFS for large files).
5. Click **"Deploy"** — Streamlit Cloud will install dependencies from
   `requirements.txt` and launch the app.

## Screenshots

> Run the app and take screenshots to add here!

| Single Analysis | Batch Analysis | Model Insights |
| --------------- | -------------- | -------------- |
| *screenshot*    | *screenshot*   | *screenshot*   |

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file
for details.
