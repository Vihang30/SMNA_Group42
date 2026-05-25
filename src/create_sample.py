"""Cuts down raw CSVs to a small sample Canvas will accept (10 MB cap)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config_loader import load_config

MAX_BYTES = 10 * 1024 * 1024


def create_sample(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    raw = Path(cfg["paths"]["raw_dir"])
    sample_dir = Path(cfg["paths"]["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(raw / "submissions.csv") if (raw / "submissions.csv").exists() else pd.DataFrame()
    com = pd.read_csv(raw / "comments.csv") if (raw / "comments.csv").exists() else pd.DataFrame()

    # a bit from each release × subreddit so markers can see the columns
    if not sub.empty:
        sub_sample = (
            sub.groupby(["release_id", "subreddit"], group_keys=False)[
                sub.columns
            ]
            .apply(lambda g: g.head(min(25, len(g))))
            .reset_index(drop=True)
        )
    else:
        sub_sample = sub

    if not com.empty and not sub_sample.empty:
        ids = set(sub_sample["id"].astype(str))
        com_sample = com[com["submission_id"].astype(str).isin(ids)].head(2000)
    elif not com.empty:
        com_sample = com.head(2000)
    else:
        com_sample = com

    sub_path = sample_dir / "submissions_sample.csv"
    com_path = sample_dir / "comments_sample.csv"
    sub_sample.to_csv(sub_path, index=False)
    com_sample.to_csv(com_path, index=False)

    readme = sample_dir / "README_sample.txt"
    readme.write_text(
        "Representative sample for Assignment 2 submission.\n"
        "Full dataset stored in data/raw/ (not submitted if too large).\n"
        "Files: submissions_sample.csv, comments_sample.csv\n"
        "Schema matches raw collection outputs.\n",
        encoding="utf-8",
    )

    total = sub_path.stat().st_size + com_path.stat().st_size
    if total > MAX_BYTES:
        # still too big — trim comments until we're under the limit
        frac = MAX_BYTES / total * 0.9
        n = max(100, int(len(com_sample) * frac))
        com_sample.head(n).to_csv(com_path, index=False)

    meta = {
        "n_submissions_sample": len(sub_sample),
        "n_comments_sample": len(com_sample),
        "size_bytes": sub_path.stat().st_size + com_path.stat().st_size,
    }
    (sample_dir / "sample_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sample_dir
