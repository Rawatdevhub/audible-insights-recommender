from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

BASE_FILE = RAW / "Audible_Catlog.csv"
ADVANCED_FILE = RAW / "Audible_Catlog_Advanced_Features.csv"
OUTPUT_FILE = PROCESSED / "audible_books_clean.csv"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\xa0", " ")
    text = re.sub(r"\\s+", " ", text).strip()
    junk = ["sorry!", "it’s rush hour", "by completing your purchase"]
    if any(token in text.lower() for token in junk):
        return ""
    return text


def parse_duration(value: object) -> float:
    """Return listening time in minutes; invalid/sentinel values become NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "-1", "nan"}:
        return np.nan
    hours = re.search(r"(\\d+)\\s*hour", text)
    minutes = re.search(r"(\\d+)\\s*minute", text)
    return (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)


def extract_genres(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    parts = []
    for item in text.split(","):
        item = re.sub(r"#?\\d[\\d,]*", "", item)
        item = re.sub(r"\\([^)]*\\)", "", item)
        item = item.strip(" #")
        if item and len(item) > 2 and item.lower() not in {"see top 100 in audible audiobooks"}:
            parts.append(item)
    return " | ".join(dict.fromkeys(parts))


def prepare_data() -> pd.DataFrame:
    base = pd.read_csv(BASE_FILE, encoding="utf-8-sig")
    advanced = pd.read_csv(ADVANCED_FILE, encoding="utf-8-sig")

    base.columns = [c.strip() for c in base.columns]
    advanced.columns = [c.strip() for c in advanced.columns]

    # Keep the richer row when the advanced catalog has the same title/author.
    df = advanced.merge(
        base[["Book Name", "Author", "Rating", "Number of Reviews", "Price"]],
        on=["Book Name", "Author"],
        how="outer",
        suffixes=("_advanced", "_base"),
    )
    for field in ["Rating", "Number of Reviews", "Price"]:
        df[field] = df[f"{field}_advanced"].combine_first(df[f"{field}_base"])

    df = df.rename(columns={"Book Name": "title", "Author": "author"})
    df["title"] = df["title"].map(clean_text)
    df["author"] = df["author"].map(clean_text)
    df["description"] = df.get("Description", "").map(clean_text)
    df["genres"] = df.get("Ranks and Genre", "").map(extract_genres)
    df["rating"] = pd.to_numeric(df["Rating"], errors="coerce").replace(-1, np.nan)
    df["review_count"] = pd.to_numeric(df["Number of Reviews"], errors="coerce").fillna(0).clip(lower=0)
    df["price"] = pd.to_numeric(df["Price"], errors="coerce").replace(-1, np.nan)
    df["listening_minutes"] = df.get("Listening Time", "").map(parse_duration)

    # Prefer rows with usable ratings, richer text, and more reviews.
    df["text_length"] = (df["title"] + " " + df["description"]).str.len()
    df = df[df["title"].ne("")].copy()
    df = df.sort_values(["rating", "text_length", "review_count"], ascending=False)
    df = df.drop_duplicates(subset=["title", "author"], keep="first")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["description"] = df["description"].replace("", "No description available.")
    df["genres"] = df["genres"].replace("", "Unknown")
    df["content"] = (df["title"] + " " + df["author"] + " " + df["genres"] + " " + df["description"]).str.lower()
    df = df[["title", "author", "rating", "review_count", "price", "listening_minutes", "genres", "description", "content"]].reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df


if __name__ == "__main__":
    data = prepare_data()
    print(f"Saved {len(data):,} cleaned books to {OUTPUT_FILE}")
