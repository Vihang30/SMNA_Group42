"""Strips URLs/markdown, drops junk rows, and builds one table for NLP."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config_loader import load_config

URL_RE = re.compile(r"http\S+|www\.\S+")
MARKDOWN_RE = re.compile(r"[*_~`>#\[\]()]+")


def clean_text(text: str | float) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = URL_RE.sub(" ", text)
    t = MARKDOWN_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = Path(cfg["paths"]["raw_dir"])
    sub_path = raw / "submissions.csv"
    com_path = raw / "comments.csv"
    if not sub_path.exists():
        raise FileNotFoundError(f"Missing {sub_path}. Run collection first.")
    submissions = pd.read_csv(sub_path)
    comments = pd.read_csv(com_path) if com_path.exists() else pd.DataFrame()
    return submissions, comments


def preprocess(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    pp = cfg["preprocessing"]
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    submissions, comments = load_raw(cfg)

    # posts: title + body
    submissions["title"] = submissions.get("title", "").fillna("").astype(str)
    submissions["selftext"] = submissions.get("selftext", "").fillna("").astype(str)
    submissions["text_raw"] = (submissions["title"] + " " + submissions["selftext"]).str.strip()
    submissions["text_clean"] = submissions["text_raw"].map(clean_text)
    submissions = submissions[submissions["text_clean"].str.len() > 10].copy()
    submissions["doc_type"] = "submission"
    submissions["author"] = submissions.get("author", "[deleted]").fillna("[deleted]")

    if comments.empty:
        print("Warning: no comments found — network analysis will be limited.")
        merged = submissions[
            ["id", "subreddit", "release_id", "phase", "author", "text_clean", "doc_type", "created_utc", "score"]
        ].rename(columns={"id": "doc_id"})
    else:
        comments["body"] = comments.get("body", "").fillna("").astype(str)
        comments["text_raw"] = comments["body"]
        comments["text_clean"] = comments["text_raw"].map(clean_text)
        comments = comments[comments["text_clean"].str.len() >= pp["min_comment_length"]].copy()
        comments["doc_type"] = "comment"
        comments["author"] = comments.get("author", "[deleted]").fillna("[deleted]")

        if pp["remove_deleted_authors"]:
            comments = comments[comments["author"] != "[deleted]"]
            submissions = submissions[submissions["author"] != "[deleted]"]

        # drop one-off accounts so the graph isn't full of isolates
        user_counts = comments["author"].value_counts()
        active = user_counts[user_counts >= pp["min_user_comments"]].index
        comments = comments[comments["author"].isin(active)]

        merged = pd.concat(
            [
                submissions[
                    ["id", "subreddit", "release_id", "phase", "author", "text_clean", "doc_type", "created_utc", "score"]
                ].rename(columns={"id": "doc_id"}),
                comments[
                    ["id", "subreddit", "release_id", "phase", "author", "text_clean", "doc_type", "created_utc", "score", "submission_id"]
                ].rename(columns={"id": "doc_id"}),
            ],
            ignore_index=True,
        )

    submissions.to_csv(out_dir / "submissions_clean.csv", index=False)
    if not comments.empty:
        comments.to_csv(out_dir / "comments_clean.csv", index=False)
    merged.to_csv(out_dir / "documents.csv", index=False)

    return {"submissions": submissions, "comments": comments, "documents": merged}
