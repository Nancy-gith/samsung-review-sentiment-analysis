"""
Samsung Galaxy Consumer Review Sentiment & Theme Analysis
------------------------------------------------------------
Analyzes 492 unique Samsung Galaxy customer reviews (Amazon.in) to surface
sentiment distribution and the specific product/service themes driving it.

Data source: msiddhu/sentiment-analysis_on_phone-reviews (public GitHub dataset,
scraped from Amazon.in). Filtered to Samsung Galaxy models only.

Author: Nancy Pandey
"""

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = "phone_reviews.csv"
OUTPUT_CSV = "samsung_reviews_final.csv"
OUTPUT_CHART = "samsung_sentiment_analysis.png"

STOPWORDS = set(
    """the a an is are was were be been being to of and or for with in on at by from this that these those
    it its i you your he she they we our my me him her them us as but if so not no very just really also too much more
    phone mobile product samsung galaxy get got have has had do does did will would can could all one good bad
    amazon even like use using time back""".split()
)

LOGISTICS_TERMS = ["return", "replacement", "delivery", "seller", "refund", "days", "received"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw review data, filter to Samsung, and clean text fields."""
    df = pd.read_csv(path)
    samsung = df[df["mobile_names"].str.contains("Samsung", case=False, na=False)].copy()
    samsung["body_clean"] = (
        samsung["body"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    samsung = samsung[samsung["body_clean"].str.len() > 10]
    # Amazon shows the same reviews on every colour/RAM variant of a phone, so
    # the scraped data repeats each review several times. Keep one copy of each.
    samsung = samsung.drop_duplicates(subset="body")
    return samsung


def label_sentiment_from_rating(star: int) -> str:
    """Use star rating as ground-truth sentiment.

    Note: an earlier pass tried VADER sentiment scoring directly on the review
    text, but the source dataset's text had already been stemmed and stripped
    of punctuation/stopwords upstream -- which destroys the negation and
    intensity cues VADER depends on (e.g. "not good" -> "good"). Validating
    against star ratings surfaced ~1,000 misclassified 1-star reviews scored
    as "Positive." Star rating is the more reliable ground truth available
    here, so it's used directly for sentiment, and text mining is used
    separately below to explain *why*.
    """
    if star <= 2:
        return "Negative"
    elif star == 3:
        return "Neutral"
    return "Positive"


def top_bigrams(text_series: pd.Series, n: int = 10):
    """Return the most common two-word phrases in a set of reviews."""
    bigrams = []
    for text in text_series.astype(str):
        tokens = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]
        bigrams.extend(zip(tokens, tokens[1:]))
    return Counter(bigrams).most_common(n)


def logistics_share(df: pd.DataFrame) -> float:
    """Share of negative reviews mentioning return/delivery/logistics terms."""
    negative = df[df["sentiment"] == "Negative"]
    mentions = negative["body"].astype(str).str.lower().apply(
        lambda t: any(term in t for term in LOGISTICS_TERMS)
    )
    return mentions.mean()


def plot_summary(df: pd.DataFrame, pos_themes: dict, neg_themes: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"])
    colors = ["#2ca02c", "#999999", "#d62728"]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title(f"Samsung Galaxy Review Sentiment\n(n={len(df):,} Amazon.in reviews)",
                       fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_ylim(0, counts.max() * 1.18)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.02, f"{v}\n({v / counts.sum() * 100:.1f}%)", ha="center", fontsize=10)

    labels = [f"{k} (pos)" for k in pos_themes] + [f"{k} (neg)" for k in neg_themes]
    values = list(pos_themes.values()) + list(neg_themes.values())
    bar_colors = ["#2ca02c"] * len(pos_themes) + ["#d62728"] * len(neg_themes)
    y_pos = np.arange(len(labels))
    axes[1].barh(y_pos, values, color=bar_colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("Most-Mentioned Themes\nby Review Sentiment", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mentions")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    df = load_and_clean(INPUT_CSV)
    df["sentiment"] = df["star"].apply(label_sentiment_from_rating)

    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print((df["sentiment"].value_counts(normalize=True) * 100).round(1))

    pos_bigrams = top_bigrams(df[df["sentiment"] == "Positive"]["body"], n=5)
    neg_bigrams = top_bigrams(df[df["sentiment"] == "Negative"]["body"], n=5)
    pos_themes = {" ".join(k): v for k, v in pos_bigrams}
    neg_themes = {" ".join(k): v for k, v in neg_bigrams}

    print("\nTop positive themes:", pos_themes)
    print("Top negative themes:", neg_themes)

    share = logistics_share(df)
    print(f"\nShare of negative reviews mentioning return/delivery/logistics: {share * 100:.1f}%")

    plot_summary(df, pos_themes, neg_themes, OUTPUT_CHART)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}, {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
