from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "audible_books_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "recommender.joblib"


def build_model() -> dict:
    books = pd.read_csv(DATA_FILE)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    matrix = vectorizer.fit_transform(books["content"].fillna(""))
    n_clusters = max(2, min(12, len(books) // 100))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    books["cluster"] = clusterer.fit_predict(matrix)
    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {"books": books, "vectorizer": vectorizer, "matrix": matrix, "clusterer": clusterer}
    joblib.dump(bundle, MODEL_FILE)
    books.to_csv(DATA_FILE, index=False)
    return bundle


def load_model() -> dict:
    if not MODEL_FILE.exists():
        if not DATA_FILE.exists():
            from src.data_pipeline import prepare_data
            prepare_data()
        return build_model()
    return joblib.load(MODEL_FILE)


def recommend(title: str, n: int = 5) -> pd.DataFrame:
    bundle = load_model()
    books, matrix = bundle["books"], bundle["matrix"]
    matches = books[books["title"].str.casefold() == title.casefold()]
    if matches.empty:
        matches = books[books["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No book found for: {title}")
    idx = matches.index[0]
    scores = linear_kernel(matrix[idx], matrix).ravel()
    quality = (books["rating"].fillna(books["rating"].median()) / 5) * 0.05
    scores = scores + quality.to_numpy()
    order = np.argsort(scores)[::-1]
    order = [i for i in order if i != idx][:n]
    result = books.iloc[order].copy()
    result["similarity"] = scores[order]
    return result[["title", "author", "rating", "review_count", "genres", "cluster", "similarity"]]


def hidden_gems(n: int = 10) -> pd.DataFrame:
    books = load_model()["books"].copy()
    reviews = np.log1p(books["review_count"])
    # High ratings, but not only the most reviewed books.
    score = (books["rating"] / 5) * 0.65 + (1 - reviews / max(reviews.max(), 1)) * 0.35
    books["hidden_gem_score"] = score
    return books.sort_values("hidden_gem_score", ascending=False).head(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        bundle = build_model()
        print(f"Built model for {len(bundle['books']):,} books")
    else:
        print(recommend("Atomic Habits", 5).to_string(index=False))
