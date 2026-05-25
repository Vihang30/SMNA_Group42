"""
Builds co-comment networks: users are nodes, edge if they commented on the same post.

Edge weight = how many threads they shared. We run Louvain + centrality for the report.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from src.config_loader import load_config


def build_co_comment_graph(comments: pd.DataFrame, cfg: dict) -> nx.Graph:
    net_cfg = cfg["network"]
    min_weight = net_cfg["min_edge_weight"]

    # need to know which thread each comment belongs to
    sub_col = "submission_id" if "submission_id" in comments.columns else None
    if sub_col is None:
        raise ValueError("comments need submission_id for co-comment network")

    by_post: dict[str, set[str]] = defaultdict(set)
    for _, row in comments.iterrows():
        author = row.get("author")
        sid = row.get(sub_col)
        if not author or author == "[deleted]" or not sid:
            continue
        by_post[str(sid)].add(author)

    edge_weights: dict[tuple[str, str], int] = defaultdict(int)
    for users in by_post.values():
        if len(users) < 2:
            continue
        for u1, u2 in combinations(sorted(users), 2):
            edge_weights[(u1, u2)] += 1

    G = nx.Graph()
    for (u1, u2), w in edge_weights.items():
        if w >= min_weight:
            G.add_edge(u1, u2, weight=w)

    # graph gets messy above ~800 nodes — keep the main blob
    max_users = net_cfg["max_users"]
    if G.number_of_nodes() > max_users:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    return G


def compute_metrics(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except (nx.NetworkXError, nx.AmbiguousSolution):
        # numpy version complains if the graph isn't one connected piece
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
        except nx.NetworkXException:
            eigenvector = {n: 0.0 for n in G.nodes()}

    clustering = nx.clustering(G, weight="weight")

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "author": node,
                "degree": degree.get(node, 0),
                "betweenness": betweenness.get(node, 0),
                "closeness": closeness.get(node, 0),
                "eigenvector": eigenvector.get(node, 0),
                "clustering": clustering.get(node, 0),
            }
        )
    return pd.DataFrame(rows).sort_values("betweenness", ascending=False)


def detect_communities(G: nx.Graph) -> dict[str, int]:
    if G.number_of_nodes() < 3:
        return {n: 0 for n in G.nodes()}
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    mapping: dict[str, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            mapping[node] = cid
    return mapping


def _graphs_for_groups(comments: pd.DataFrame, cfg: dict, group_col: str | None) -> dict[str, nx.Graph]:
    graphs: dict[str, nx.Graph] = {}
    if group_col is None or group_col not in comments.columns:
        if len(comments) >= 30:
            graphs["all"] = build_co_comment_graph(comments, cfg)
        return graphs
    for label, grp in comments.groupby(group_col):
        if len(grp) < 30:
            continue
        graphs[str(label)] = build_co_comment_graph(grp, cfg)
    return graphs


def network_by_phase(comments: pd.DataFrame, cfg: dict) -> dict[str, nx.Graph]:
    return _graphs_for_groups(comments, cfg, "phase")


def network_by_release(comments: pd.DataFrame, cfg: dict) -> dict[str, nx.Graph]:
    return _graphs_for_groups(comments, cfg, "release_id")


def run_network(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    tab_dir = Path(cfg["paths"]["tables_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    com_path = proc / "comments_clean.csv"
    if not com_path.exists():
        raise FileNotFoundError("Run preprocess first to create comments_clean.csv")

    comments = pd.read_csv(com_path)
    phase_graphs = network_by_phase(comments, cfg)
    release_graphs = network_by_release(comments, cfg)

    summary_rows = []
    all_metrics = []

    for phase, G in phase_graphs.items():
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0
        try:
            modularity = nx.community.modularity(
                G, nx.community.louvain_communities(G, weight="weight", seed=42), weight="weight"
            )
        except Exception:
            modularity = float("nan")

        summary_rows.append(
            {
                "group_type": "phase",
                "group": phase,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_clustering": nx.average_clustering(G, weight="weight") if n_edges else 0,
                "modularity": modularity,
            }
        )

        comm_map = detect_communities(G) if n_nodes else {}
        metrics = compute_metrics(G)
        if not metrics.empty:
            metrics["phase"] = phase
            metrics["community"] = metrics["author"].map(comm_map)
            all_metrics.append(metrics)

        if 0 < n_nodes <= 400:
            _draw_network(G, comm_map, fig_dir / f"network_{phase}.png", phase)

    for release_id, G in release_graphs.items():
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0
        try:
            modularity = nx.community.modularity(
                G, nx.community.louvain_communities(G, weight="weight", seed=42), weight="weight"
            )
        except Exception:
            modularity = float("nan")
        summary_rows.append(
            {
                "group_type": "release",
                "group": release_id,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_clustering": nx.average_clustering(G, weight="weight") if n_edges else 0,
                "modularity": modularity,
            }
        )

    pd.DataFrame(summary_rows).to_csv(tab_dir / "network_summary.csv", index=False)
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        metrics_df.to_csv(tab_dir / "centrality_top_users.csv", index=False)
        top = metrics_df.groupby("phase").head(15)
        top.to_csv(tab_dir / "influential_users_by_phase.csv", index=False)

    # bar charts for the write-up
    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        phase_summ = summ[summ["group_type"] == "phase"]
        if not phase_summ.empty:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=phase_summ, x="group", y="density", hue="group", legend=False)
            plt.title("Co-comment network density by phase")
            plt.tight_layout()
            plt.savefig(fig_dir / "network_density_by_phase.png", dpi=150)
            plt.close()
        release_summ = summ[summ["group_type"] == "release"]
        if not release_summ.empty:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=release_summ, x="group", y="modularity", hue="group", legend=False)
            plt.title("Louvain modularity by release")
            plt.tight_layout()
            plt.savefig(fig_dir / "network_modularity_by_release.png", dpi=150)
            plt.close()

    return {"phase_graphs": phase_graphs, "release_graphs": release_graphs, "summary": pd.DataFrame(summary_rows)}


def _draw_network(G: nx.Graph, communities: dict, out_path: Path, title: str) -> None:
    if G.number_of_nodes() == 0:
        return
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.4)
    colors = [communities.get(n, 0) for n in G.nodes()]
    sizes = [80 + 40 * G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab10, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    plt.title(f"User co-comment network — {title}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
