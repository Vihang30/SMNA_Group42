"""
Grabs posts and comments from PullPush (archived Reddit).

No Reddit login needed — just be polite with the delay in config and mention
PullPush + your date range in the report.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from src.config_loader import load_config


def _epoch_window(announcement_utc: int, pre_days: int, post_days: int) -> tuple[int, int]:
    pre = announcement_utc - pre_days * 86400
    post = announcement_utc + post_days * 86400
    return pre, post


def _fetch_page(
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    delay: float,
) -> list[dict]:
    """Keeps requesting pages until we hit max_total or run out of results."""
    url = f"{base_url}/{endpoint}/"
    all_rows: list[dict] = []
    size = min(int(params.get("size", 100)), 100)
    max_total = int(params.get("max_total", 500))
    before = params.get("before")

    while len(all_rows) < max_total:
        p = {k: v for k, v in params.items() if k not in ("after", "before", "max_total", "size")}
        p["size"] = size
        p["sort"] = "desc"
        p["sort_type"] = "created_utc"
        if params.get("after") is not None:
            p["after"] = params["after"]
        if before is not None:
            p["before"] = before
        resp = requests.get(url, params=p, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            break
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        if len(data) < size:
            break
        # walk backwards in time for the next page
        oldest = min(row.get("created_utc", 0) for row in data)
        before = int(oldest) - 1
        time.sleep(delay)
    return all_rows[:max_total]


def collect_submissions(
    cfg: dict,
    release: dict,
    subreddit: str,
    query: str,
) -> pd.DataFrame:
    coll = cfg["collection"]
    pre, post = _epoch_window(
        release["announcement_utc"], release["pre_days"], release["post_days"]
    )
    params = {
        "subreddit": subreddit,
        "q": query,
        "after": pre,
        "before": post,
        "size": 100,
        "max_total": coll["submissions_per_query"],
    }
    rows = _fetch_page(coll["pullpush_base"], "submission", params, coll["request_delay_seconds"])
    for r in rows:
        r["release_id"] = release["id"]
        r["search_query"] = query
        r["phase"] = _phase_label(r.get("created_utc"), release)
    return pd.DataFrame(rows)


def _phase_label(created_utc: int | None, release: dict) -> str:
    """pre = before launch, peak = first 3 days, post = rest of the window."""
    if created_utc is None:
        return "unknown"
    ann = release["announcement_utc"]
    if created_utc < ann:
        return "pre"
    if created_utc < ann + 3 * 86400:  # 3 days of hype
        return "peak"
    return "post"


def collect_comments_for_submissions(
    cfg: dict,
    submissions: pd.DataFrame,
) -> pd.DataFrame:
    coll = cfg["collection"]
    base = coll["pullpush_base"]
    delay = coll["request_delay_seconds"]
    limit = coll["comments_per_submission"]
    frames: list[pd.DataFrame] = []

    for _, row in tqdm(submissions.iterrows(), total=len(submissions), desc="comments"):
        link_id = row.get("id")
        if not link_id:
            continue
        params = {
            "link_id": link_id,
            "size": 100,
            "max_total": limit,
        }
        try:
            comments = _fetch_page(base, "comment", params, delay)
        except requests.RequestException:
            continue
        for c in comments:
            c["submission_id"] = link_id
            c["subreddit"] = row.get("subreddit")
            c["release_id"] = row.get("release_id")
            c["phase"] = row.get("phase")
        if comments:
            frames.append(pd.DataFrame(comments))
    if not frames:
        return pd.DataFrame()
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_collection(
    cfg: dict | None = None,
    skip_comments: bool = False,
    comments_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    coll = cfg["collection"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    sub_path = raw_dir / "submissions.csv"

    if comments_only and sub_path.exists():
        submissions = pd.read_csv(sub_path)
        com_path = raw_dir / "comments.csv"
        if not skip_comments:
            max_for_comments = int(coll.get("max_submissions_for_comments", 250))
            to_fetch = submissions.sort_values("num_comments", ascending=False).head(max_for_comments)
            comments = collect_comments_for_submissions(cfg, to_fetch)
            if not comments.empty:
                comments = comments.drop_duplicates(subset=["id"], keep="first")
                comments.to_csv(com_path, index=False)
                print(f"Saved {len(comments)} comments -> {com_path}")
        comments = pd.read_csv(com_path) if com_path.exists() else pd.DataFrame()
        meta = {
            "source": "PullPush API",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "n_submissions": len(submissions),
            "n_comments": len(comments),
            "mode": "comments_only",
        }
        (raw_dir / "collection_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return submissions, comments

    sub_frames: list[pd.DataFrame] = []
    for release in cfg["releases"]:
        for sub in cfg["subreddits"]:
            for q in release["search_queries"]:
                print(f"Collecting submissions: {release['id']} | r/{sub} | {q}")
                try:
                    df = collect_submissions(cfg, release, sub, q)
                    if not df.empty:
                        sub_frames.append(df)
                except requests.RequestException as exc:
                    print(f"  Warning: {exc}")

    if not sub_frames:
        raise RuntimeError("No submissions collected. Check network or API availability.")

    submissions = pd.concat(sub_frames, ignore_index=True)
    submissions = submissions.drop_duplicates(subset=["id"], keep="first")
    submissions["collected_at"] = datetime.now(timezone.utc).isoformat()

    submissions.to_csv(sub_path, index=False)
    print(f"Saved {len(submissions)} submissions -> {sub_path}")

    comments = pd.DataFrame()
    max_for_comments = int(coll.get("max_submissions_for_comments", 250))
    if not skip_comments and len(submissions) > 0:
        to_fetch = submissions.sort_values("num_comments", ascending=False).head(max_for_comments)
        comments = collect_comments_for_submissions(cfg, to_fetch)
        if not comments.empty:
            comments = comments.drop_duplicates(subset=["id"], keep="first")
            com_path = raw_dir / "comments.csv"
            comments.to_csv(com_path, index=False)
            print(f"Saved {len(comments)} comments -> {com_path}")

    meta = {
        "source": "PullPush API",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "n_submissions": len(submissions),
        "n_comments": len(comments) if not comments.empty else 0,
        "subreddits": cfg["subreddits"],
        "releases": [r["id"] for r in cfg["releases"]],
    }
    (raw_dir / "collection_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return submissions, comments


if __name__ == "__main__":
    run_collection()
