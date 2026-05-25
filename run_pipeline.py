#!/usr/bin/env python
"""
Main script — runs collection (optional) then preprocessing, NLP, and network steps.

  python run_pipeline.py                    # analyse whatever is in data/raw/
  python run_pipeline.py --collect            # scrape via PullPush (takes a while)
  python run_pipeline.py --collect-praw       # needs .env with Reddit keys
  python run_pipeline.py --comments-only      # posts already saved, just fetch comments
  python run_pipeline.py --skip-collect       # skip scraping, analysis only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.create_sample import create_sample
from src.network_analysis import run_network
from src.nlp_analysis import run_nlp
from src.preprocess import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Reddit communities — analysis pipeline")
    parser.add_argument("--collect", action="store_true", help="Collect data via PullPush API")
    parser.add_argument("--collect-praw", action="store_true", help="Collect data via PRAW")
    parser.add_argument("--skip-comments", action="store_true", help="Skip comment collection")
    parser.add_argument("--skip-collect", action="store_true", help="Only run analysis on existing raw data")
    parser.add_argument(
        "--comments-only",
        action="store_true",
        help="Fetch comments for existing data/raw/submissions.csv",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw_sub = Path(cfg["paths"]["raw_dir"]) / "submissions.csv"

    if args.collect_praw:
        from src.collect_praw import run_praw_collection

        run_praw_collection(cfg)
    elif args.collect or args.comments_only:
        from src.collect_pullpush import run_collection

        run_collection(cfg, skip_comments=args.skip_comments, comments_only=args.comments_only)
    elif not args.skip_collect and not raw_sub.exists():
        print("Nothing in data/raw yet — starting PullPush scrape (usually 30–60 min)...")
        from src.collect_pullpush import run_collection

        run_collection(cfg, skip_comments=args.skip_comments)
    elif args.skip_collect and not raw_sub.exists():
        raise SystemExit(f"Missing {raw_sub}. Run with --collect first.")

    print("\n=== Preprocessing ===")
    preprocess(cfg)

    print("\n=== NLP (sentiment + LDA topics) ===")
    run_nlp(cfg)

    print("\n=== Network analysis ===")
    run_network(cfg)

    print("\n=== Creating submission sample ===")
    sample_dir = create_sample(cfg)
    print(f"Sample data: {sample_dir}")

    print("\n=== Done ===")
    print(f"Figures: {cfg['paths']['figures_dir']}")
    print(f"Tables:  {cfg['paths']['tables_dir']}")
    print("Report is separate — draft when you're happy with the figures.")


if __name__ == "__main__":
    main()
