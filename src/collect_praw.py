"""
Same idea as PullPush but through Reddit's official API (PRAW).

Copy .env.example to .env and fill in your app credentials.
Handy if PullPush is down or you want more recent threads.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv
from tqdm import tqdm

from src.config_loader import load_config
from src.collect_pullpush import _epoch_window, _phase_label


def _reddit_client() -> praw.Reddit:
    load_dotenv()
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def run_praw_collection(cfg: dict | None = None, max_posts_per_query: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    reddit = _reddit_client()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    posts: list[dict] = []
    comments: list[dict] = []

    for release in cfg["releases"]:
        pre, post = _epoch_window(
            release["announcement_utc"], release["pre_days"], release["post_days"]
        )
        for sub_name in cfg["subreddits"]:
            sub = reddit.subreddit(sub_name)
            for query in release["search_queries"]:
                for submission in sub.search(query, sort="new", time_filter="all", limit=max_posts_per_query):
                    created = int(submission.created_utc)
                    if created < pre or created > post:
                        continue
                    posts.append(
                        {
                            "id": submission.id,
                            "subreddit": sub_name,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "created_utc": created,
                            "author": str(submission.author) if submission.author else "[deleted]",
                            "permalink": submission.permalink,
                            "release_id": release["id"],
                            "search_query": query,
                            "phase": _phase_label(created, release),
                        }
                    )
                    submission.comments.replace_more(limit=0)
                    for c in submission.comments.list()[: cfg["collection"]["comments_per_submission"]]:
                        comments.append(
                            {
                                "id": c.id,
                                "submission_id": submission.id,
                                "subreddit": sub_name,
                                "body": c.body,
                                "score": c.score,
                                "created_utc": int(c.created_utc),
                                "author": str(c.author) if c.author else "[deleted]",
                                "parent_id": c.parent_id,
                                "release_id": release["id"],
                                "phase": _phase_label(int(c.created_utc), release),
                            }
                        )

    submissions = pd.DataFrame(posts).drop_duplicates(subset=["id"])
    comments_df = pd.DataFrame(comments).drop_duplicates(subset=["id"])
    submissions["collected_at"] = datetime.now(timezone.utc).isoformat()

    submissions.to_csv(raw_dir / "submissions.csv", index=False)
    if not comments_df.empty:
        comments_df.to_csv(raw_dir / "comments.csv", index=False)
    return submissions, comments_df


if __name__ == "__main__":
    run_praw_collection()
