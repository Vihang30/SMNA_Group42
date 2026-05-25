# How do AI Communities react to new AI releases?

**Course:** Social Media and Network Analysis (COSC 2671 / COSC 3047) — Assignment 2  
**Topic:** Reddit reactions to Llama 3, GPT-4o, and Claude 3.5 Sonnet  
**Methods:** VADER sentiment, LDA topics, co-comment networks (NetworkX)

---

## What this project does

We look at how different AI subreddits talk when a big model drops—not just whether posts are positive or negative, but also **what people discuss** and **who ends up in the same threads**.

**Research question:** How do r/MachineLearning, r/LocalLLaMA, r/ChatGPT, r/artificial, and r/OpenAI differ in sentiment, topics, and user interaction across **pre**, **peak** (launch + 3 days), and **post** release windows?

Data comes from Reddit via the **PullPush** archive API (no Reddit login needed for the main pipeline). There is an optional **PRAW** path if we have API keys.

---

## Folder layout

```
Social_Media_A2/
├── config.yaml           # subreddits, release dates, collection limits
├── run_pipeline.py       # main script — run this
├── requirements.txt
├── src/                  # collection, cleaning, NLP, networks
├── notebooks/
│   └── 01_run_analysis.ipynb   # same workflow, step by step
├── data/
│   ├── raw/              # full scrape (not submitted — too big)
│   ├── processed/        # rebuilt when you run the pipeline
│   └── sample/           # small CSVs for Canvas (under 10 MB, hopefully)
└── outputs/
    ├── figures/          # charts for the report
    └── tables/           # CSV results
```

---

## Getting started

```bash
cd Social_Media_A2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you use PRAW instead of PullPush: copy `.env.example` to `.env` and add your Reddit app credentials. Do not commit `.env`.

---

## How to run everything

### First time — collect data

PullPush (what we used for the report):

```bash
python run_pipeline.py --collect
```

This can take 30–60 minutes because of API rate limits. Submissions are saved first; comments are fetched for the busiest threads.

Already have `data/raw/submissions.csv` but need more comments?

```bash
python run_pipeline.py --comments-only
```

Quick test without comments (network analysis will be weak):

```bash
python run_pipeline.py --collect --skip-comments
```

### Analysis only (data already in `data/raw/`)

```bash
python run_pipeline.py --skip-collect
```

That runs preprocessing → VADER + LDA → networks → builds `data/sample/`.

### Optional: Jupyter (VS Code / Cursor)

```bash
pip install -r requirements-notebook.txt
```

In the notebook: **Select Kernel** → **Python (SMNA A2)** (`C:\smna-venv` if you used the short-path install). Open the **`Social_Media_A2`** folder (not the `.ipynb` alone). Keep `COLLECT = False` if you have `data/raw/` locally; for submission-only copies use `python run_pipeline.py` instead.

---

## What you get in `outputs/`

| File | What it is |
|------|------------|
| `outputs/figures/sentiment_*.png` | Sentiment by release, subreddit, phase |
| `outputs/figures/network_*.png` | Co-comment network graphs |
| `outputs/tables/sentiment_by_release_phase_subreddit.csv` | Mean compound, % positive/negative |
| `outputs/tables/lda_topics.csv` | Top words per topic |
| `outputs/tables/network_summary.csv` | Nodes, edges, density, modularity |
| `outputs/tables/influential_users_by_phase.csv` | Users with high betweenness |
| `data/sample/` | Stratified sample for submission |

---

## Network (for the report)

| | |
|--|--|
| **Nodes** | Reddit usernames from comments |
| **Edges** | Two users commented on the same post (undirected) |
| **Weight** | How many posts they shared |
| **Communities** | Louvain |
| **Metrics** | Degree, betweenness, eigenvector, clustering, modularity |

We drop very weak edges via `min_edge_weight` in `config.yaml`. Large graphs keep the biggest connected piece for plots.

---

## NLP (for the report)

- **VADER** — sentiment on cleaned post/comment text  
- **LDA (8 topics)** — main themes in the corpus  

Settings live in `config.yaml` under `nlp:`.

---

## Data notes

- **Source:** [PullPush](https://api.pullpush.io/) (primary) or PRAW (optional)  
- **Our run:** 836 submissions, 1,327 comments → 1,970 documents after cleaning  
- **Submitted to Canvas:** `data/sample/` only — not full `data/raw/`  
- **Ethics:** Public Reddit data only; no keys or private info in the repo  

Sampling is not perfect (keyword search, comment caps per thread). We explain that in the report.

---

## Changing settings

Edit `config.yaml` if you want different subreddits, release dates, `pre_days` / `post_days`, how many posts to scrape, or network/LDA parameters.

---

## Submission reminder

| Include | Do not include |
|---------|----------------|
| Code, `config.yaml`, `requirements.txt`, this README | `.env`, `venv/` |
| `data/sample/` | Full `data/raw/` (gitignored) |
| `outputs/figures/`, `outputs/tables/` | `data/processed/` (regenerated on run) |
| Report PDF + Access `.txt` + Worksheet (Canvas) | Assignment spec PDF |

Rename files with your student number and group per Canvas (e.g. `Report_s4149812_UG_Group_12.pdf`).

---

## Reproducing our report figures

With the same `data/raw/` files (or after `--collect`):

```bash
python run_pipeline.py --skip-collect
```

Figures in `outputs/figures/` should match what we used in `Report_UG_Group_XX.docx`.

---

## Team

1. Vihang Mehere - s4149812
2. Siddhant Kripal - s4184755
