"""VADER sentiment + LDA topics, saves CSVs and a few charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config_loader import load_config

NLTK_PACKAGES = ("stopwords", "punkt", "punkt_tab")


def _ensure_nltk() -> None:
    for pkg in NLTK_PACKAGES:
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def add_sentiment(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    _ensure_nltk()
    sia = SentimentIntensityAnalyzer()
    scores = df[text_col].fillna("").map(lambda t: sia.polarity_scores(t))
    out = df.copy()
    out["sentiment_compound"] = scores.map(lambda s: s["compound"])
    out["sentiment_pos"] = scores.map(lambda s: s["pos"])
    out["sentiment_neg"] = scores.map(lambda s: s["neg"])
    out["sentiment_neu"] = scores.map(lambda s: s["neu"])
    out["sentiment_label"] = out["sentiment_compound"].map(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    return out


def run_lda(df: pd.DataFrame, cfg: dict, text_col: str = "text_clean") -> tuple[pd.DataFrame, list[list[tuple[str, float]]]]:
    _ensure_nltk()
    from nltk.corpus import stopwords

    n_topics = cfg["nlp"]["n_topics"]
    stop = set(stopwords.words("english"))

    vectorizer = CountVectorizer(
        max_features=cfg["nlp"]["max_features_lda"],
        stop_words=list(stop),
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    texts = df[text_col].fillna("").tolist()
    if len(texts) < 30:
        return df, []

    min_df = 3 if len(texts) < 200 else 5
    vectorizer.set_params(min_df=min_df)

    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=cfg["nlp"]["random_state"],
        max_iter=20,
    )
    doc_topics = lda.fit_transform(dtm)
    topic_ids = doc_topics.argmax(axis=1)
    out = df.copy()
    out["topic_id"] = topic_ids
    out["topic_strength"] = doc_topics.max(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    top_words: list[list[tuple[str, float]]] = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-12:][::-1]
        top_words.append([(feature_names[i], float(topic[i])) for i in top_indices])

    return out, top_words


def aggregate_sentiment(sent_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["release_id", "phase", "subreddit"]
    available = [c for c in group_cols if c in sent_df.columns]
    agg = (
        sent_df.groupby(available, dropna=False)
        .agg(
            n_docs=("sentiment_compound", "count"),
            mean_compound=("sentiment_compound", "mean"),
            pct_positive=("sentiment_label", lambda s: (s == "positive").mean()),
            pct_negative=("sentiment_label", lambda s: (s == "negative").mean()),
        )
        .reset_index()
    )
    return agg


def run_nlp(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    tab_dir = Path(cfg["paths"]["tables_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(proc / "documents.csv")
    docs = add_sentiment(docs)
    docs, top_words = run_lda(docs, cfg)

    docs.to_csv(proc / "documents_with_nlp.csv", index=False)
    agg = aggregate_sentiment(docs)
    agg.to_csv(tab_dir / "sentiment_by_release_phase_subreddit.csv", index=False)

    # write out what each LDA topic looks like in words
    if top_words:
        rows = []
        for tid, words in enumerate(top_words):
            label = ", ".join(w for w, _ in words[:6])
            rows.append({"topic_id": tid, "top_terms": label})
        pd.DataFrame(rows).to_csv(tab_dir / "lda_topics.csv", index=False)

    # chart: sentiment across releases and pre/peak/post
    if "phase" in docs.columns and "release_id" in docs.columns:
        plt.figure(figsize=(10, 5))
        plot_df = docs.groupby(["release_id", "phase"], as_index=False)["sentiment_compound"].mean()
        sns.barplot(data=plot_df, x="release_id", y="sentiment_compound", hue="phase")
        plt.title("Mean sentiment compound by release and phase")
        plt.ylabel("VADER compound score")
        plt.tight_layout()
        plt.savefig(fig_dir / "sentiment_by_release_phase.png", dpi=150)
        plt.close()

    # chart: does r/ChatGPT feel different from r/LocalLLaMA?
    if "subreddit" in docs.columns:
        plt.figure(figsize=(11, 5))
        sub_agg = docs.groupby(["subreddit", "phase"], as_index=False)["sentiment_compound"].mean()
        sns.barplot(data=sub_agg, x="subreddit", y="sentiment_compound", hue="phase")
        plt.title("Mean sentiment by subreddit and phase")
        plt.tight_layout()
        plt.savefig(fig_dir / "sentiment_by_subreddit_phase.png", dpi=150)
        plt.close()

    # chart: how many docs landed in each topic
    if "topic_id" in docs.columns:
        plt.figure(figsize=(10, 5))
        topic_counts = docs["topic_id"].value_counts().sort_index()
        topic_counts.plot(kind="bar")
        plt.title("Document count per LDA topic")
        plt.xlabel("Topic ID")
        plt.tight_layout()
        plt.savefig(fig_dir / "topic_distribution.png", dpi=150)
        plt.close()

    return {"documents": docs, "sentiment_agg": agg, "lda_topics": top_words}
