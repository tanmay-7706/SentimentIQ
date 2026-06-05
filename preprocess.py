"""
preprocess.py — Text preprocessing pipeline for SentimentIQ.

Provides a TextPreprocessor class that cleans, normalises, and stems
raw movie-review text before it is fed into TF-IDF vectorisers.
"""

import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------

def download_nltk_resources() -> None:
    """Silently download every NLTK resource the pipeline needs."""
    for resource in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"):
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """End-to-end text cleaning pipeline for sentiment analysis."""

    def __init__(self) -> None:
        download_nltk_resources()
        self._stop_words: set[str] = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()
        # Pre-compile frequently used patterns
        self._re_url = re.compile(r"https?://\S+|www\.\S+")
        self._re_numbers = re.compile(r"\d+")
        self._re_whitespace = re.compile(r"\s+")

    # ---- individual steps ------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Lowercase, strip HTML, URLs, punctuation, numbers, whitespace."""
        text = text.lower()
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove URLs
        text = self._re_url.sub("", text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove numbers
        text = self._re_numbers.sub("", text)
        # Collapse and strip whitespace
        text = self._re_whitespace.sub(" ", text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        """Drop English stopwords."""
        return " ".join(
            word for word in text.split() if word not in self._stop_words
        )

    def stem_text(self, text: str) -> str:
        """Apply Porter stemming to every token."""
        return " ".join(self._stemmer.stem(word) for word in text.split())

    def full_pipeline(self, text: str) -> str:
        """Run clean → stopwords → stem in order."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "<br/>This is a GREAT movie!!! Visit https://example.com for more. "
        "Rating: 10/10. I absolutely loved it."
    )
    pp = TextPreprocessor()
    print("Raw  :", sample)
    print("Clean:", pp.full_pipeline(sample))
