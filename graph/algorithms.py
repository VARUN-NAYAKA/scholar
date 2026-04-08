"""
ScholAR - Graph Analysis Algorithms.
PageRank, Community Detection, Centrality, and more.
"""

import networkx as nx
import community as community_louvain
from collections import Counter
from rich.console import Console

console = Console()


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """
    Detect communities using Louvain algorithm.

    Args:
        graph: Undirected NetworkX graph

    Returns:
        Dict mapping node_id -> community_id
    """
    # Filter to paper nodes only
    paper_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "paper"]
    subgraph = graph.subgraph(paper_nodes)

    if len(subgraph) < 2:
        return {n: 0 for n in paper_nodes}

    partition = community_louvain.best_partition(subgraph, random_state=42)

    num_communities = len(set(partition.values()))
    console.print(f"[magenta]Analysis:[/magenta] Detected {num_communities} communities")

    return partition


def compute_pagerank(graph: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    """
    Compute PageRank for all nodes.

    Args:
        graph: Directed NetworkX graph
        alpha: Damping factor

    Returns:
        Dict mapping node_id -> pagerank score
    """
    if len(graph) == 0:
        return {}

    try:
        pr = nx.pagerank(graph, alpha=alpha, max_iter=100, tol=1e-6)
        console.print(f"[magenta]Analysis:[/magenta] PageRank computed for {len(pr)} nodes")
        return pr
    except nx.NetworkXError as e:
        console.print(f"[yellow]PageRank error: {e}[/yellow]")
        return {n: 1.0 / len(graph) for n in graph.nodes()}


def compute_betweenness_centrality(graph: nx.Graph, k: int = None) -> dict[str, float]:
    """
    Compute betweenness centrality (finds bridge papers).

    Args:
        graph: NetworkX graph
        k: Number of sample nodes for approximation (None = exact)

    Returns:
        Dict mapping node_id -> centrality score
    """
    if len(graph) == 0:
        return {}

    # For large graphs, use approximation
    if k is None and len(graph) > 500:
        k = min(100, len(graph))

    try:
        bc = nx.betweenness_centrality(graph, k=k, normalized=True)
        console.print(f"[magenta]Analysis:[/magenta] Betweenness centrality computed")
        return bc
    except Exception as e:
        console.print(f"[yellow]Betweenness centrality error: {e}[/yellow]")
        return {}


def compute_degree_centrality(graph: nx.Graph) -> dict[str, float]:
    """Compute degree centrality for all nodes."""
    if len(graph) == 0:
        return {}
    return nx.degree_centrality(graph)


def find_key_papers(
    graph: nx.DiGraph,
    pagerank: dict[str, float],
    betweenness: dict[str, float],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Find the most important papers combining PageRank and betweenness.

    Returns:
        List of (paper_id, combined_score) sorted by importance
    """
    paper_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "paper"]

    # Normalize scores
    pr_values = [pagerank.get(n, 0) for n in paper_nodes]
    bc_values = [betweenness.get(n, 0) for n in paper_nodes]

    max_pr = max(pr_values) if pr_values else 1
    max_bc = max(bc_values) if bc_values else 1

    combined = []
    for node in paper_nodes:
        pr_norm = pagerank.get(node, 0) / max_pr if max_pr > 0 else 0
        bc_norm = betweenness.get(node, 0) / max_bc if max_bc > 0 else 0
        score = 0.6 * pr_norm + 0.4 * bc_norm
        combined.append((node, score))

    combined.sort(key=lambda x: x[1], reverse=True)

    console.print(f"[magenta]Analysis:[/magenta] Top {min(top_k, len(combined))} key papers identified")
    return combined[:top_k]


def get_community_labels(
    graph: nx.Graph,
    partition: dict[str, int],
    paper_data: dict,
) -> dict[int, dict]:
    """
    Generate labels/summaries for each community based on paper metadata.

    Args:
        graph: The graph
        partition: community assignments
        paper_data: Dict mapping paper_id -> Paper object

    Returns:
        Dict mapping community_id -> {papers: [...], top_keywords: [...], year_range: ...}
    """
    communities: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    result = {}
    for comm_id, paper_ids in communities.items():
        papers_in_community = [paper_data[pid] for pid in paper_ids if pid in paper_data]

        if not papers_in_community:
            continue

        # Collect all topics/keywords
        all_topics = []
        years = []
        for p in papers_in_community:
            all_topics.extend(p.topics)
            all_topics.extend(p.fields_of_study)
            if p.year:
                years.append(p.year)

        # Get most common topics
        topic_counts = Counter(all_topics)
        top_topics = [t for t, _ in topic_counts.most_common(5)]

        year_range = f"{min(years)}-{max(years)}" if years else "N/A"

        result[comm_id] = {
            "paper_ids": paper_ids,
            "paper_count": len(paper_ids),
            "top_keywords": top_topics,
            "year_range": year_range,
            "avg_citations": sum(p.citation_count for p in papers_in_community) / len(papers_in_community),
        }

    return result


def detect_temporal_trends(
    papers: list,
    min_year: int = None,
) -> dict[int, int]:
    """
    Count publications per year.

    Returns:
        Dict mapping year -> paper count
    """
    year_counts: dict[int, int] = {}
    for paper in papers:
        if paper.year:
            if min_year and paper.year < min_year:
                continue
            year_counts[paper.year] = year_counts.get(paper.year, 0) + 1

    return dict(sorted(year_counts.items()))
