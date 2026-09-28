# Samsung Galaxy Consumer Review Sentiment & Theme Analysis

Analysis of **492 unique Samsung Galaxy customer reviews** from Amazon.in, uncovering
what actually drives positive vs. negative sentiment — beyond a simple polarity score.

![Sentiment Analysis Chart](samsung_sentiment_analysis.png)

## Why this project

Most beginner sentiment-analysis projects stop at "X% positive, Y% negative." This one goes
further: it identifies *which specific product and service themes* are driving that sentiment,
the way a Consumer Insights analyst would frame it for a product or marketing team.

## Key Findings

- Sentiment is nearly split: **46% positive vs. 45% negative** — a polarized customer base,
  not uniform satisfaction.
- **Camera quality** and **battery** are among the most-discussed features in *both* positive and
  negative reviews — they're the biggest expectation-setters, not universally loved or hated.
- **~49% of negative reviews** mention return, replacement, refund or delivery language — a meaningful
  share of dissatisfaction is tied to the post-purchase experience, not only the device itself.
  (This is a keyword match, so it's a rough signal: e.g. "received a defective phone" also counts.)
- "Heating issue" and "customer care" are distinct, recurring complaint clusters specific enough
  to route to product and support teams respectively.

Full write-up: [insight_summary.md](insight_summary.md)

## Methodology note (and a deliberate correction)

The first approach used VADER sentiment scoring directly on review text. Validating that output
against actual star ratings surfaced a problem: ~1,000 one-star reviews were being scored as
"Positive." The cause was the source dataset's text had already been stemmed and stripped of
punctuation/stopwords, which destroys the negation and intensity cues VADER relies on (e.g.
"not good" becomes just "good" once "not" is removed as a stopword).

Since star ratings were available as reliable ground truth, sentiment was reassigned from star
rating directly (1-2★ Negative, 3★ Neutral, 4-5★ Positive), and text mining (bigram frequency)
was used separately to explain *why* — which is arguably a better-designed approach regardless.

## Data-quality fix: duplicate reviews

A later review of the data found that the raw Samsung subset had **7,913 rows but only 492 unique
review texts** (~94% duplicates). Amazon shows the same reviews on every colour/RAM variant of a
phone, so the scraper collected each review several times. The pipeline now removes duplicates
(`drop_duplicates(subset="body")`) before any analysis. The headline findings held up: the
sentiment split moved from 44/46 to 46/45 and the main themes stayed the same.

## Data

- **Source:** [msiddhu/sentiment-analysis_on_phone-reviews](https://github.com/msiddhu/sentiment-analysis_on_phone-reviews)
  (public dataset, reviews scraped from Amazon.in)
- **Scope:** Filtered to Samsung Galaxy models only — 492 unique reviews across 6 models
  (Galaxy M01, M01 Core, M21, M31, M31s, Z Flip)

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Regex-based text mining

## Run it yourself

```bash
pip install pandas numpy matplotlib
python sentiment_analysis.py
```

Requires `phone_reviews.csv` in the same directory (see Data section for source).

## Files

- `sentiment_analysis.py` — full pipeline: load, clean, label, mine themes, chart
- `insight_summary.md` — the business-facing write-up
- `samsung_sentiment_analysis.png` — output chart
