"""
ScholAR - Knowledge Graph Builder.
Constructs a NetworkX graph from collected papers with multiple edge types.
"""

import networkx as nx
from typing import Optional
from core.models import Paper
from core.config import SIMILARITY_THRESHOLD
from rich.console import Console
import numpy as np

console = Console()


class KnowledgeGraphBuilder:
    """Builds and manages the research knowledge graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.papers: dict[str, Paper] = {}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def add_papers(self, papers: list[Paper]) -> int:
        """
        Add papers as nodes to the graph.
        Returns number of new papers added.
        """
        added = 0
        for paper in papers:
            if paper.paper_id not in self.papers:
                self.papers[paper.paper_id] = paper
                self.graph.add_node(
                    paper.paper_id,
                    type="paper",
                    title=paper.title,
                    year=paper.year,
                    citation_count=paper.citation_count,
                    venue=paper.venue or "",
                    relevance_score=paper.relevance_score,
                    abstract=paper.abstract[:200] if paper.abstract else "",
                )
                added += 1

        console.print(f"[cyan]Graph:[/cyan] Added {added} new paper nodes (total: {self.num_nodes})")
        return added

    def build_citation_edges(self):
        """Create citation edges between papers that are both in the graph."""
        edge_count = 0
        for paper_id, paper in self.papers.items():
            for ref_id in paper.references:
                if ref_id in self.papers:
                    self.graph.add_edge(
                        paper_id,
                        ref_id,
                        type="cites",
                        weight=1.0,
                    )
                    edge_count += 1

            for cite_id in paper.citations:
                if cite_id in self.papers:
                    self.graph.add_edge(
                        cite_id,
                        paper_id,
                        type="cites",
                        weight=1.0,
                    )
                    edge_count += 1

        console.print(f"[cyan]Graph:[/cyan] Built {edge_count} citation edges")
        return edge_count

    def build_similarity_edges(self, embeddings: dict[str, list[float]], threshold: float = None):
        """
        Create similarity edges between papers based on embedding cosine similarity.

        Args:
            embeddings: Dict mapping paper_id -> embedding vector
            threshold: Min cosine similarity to create an edge (default from config)
        """
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        paper_ids = [pid for pid in embeddings if pid in self.papers]
        if len(paper_ids) < 2:
            return 0

        # Build embedding matrix
        matrix = np.array([embeddings[pid] for pid in paper_ids])

        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / norms

        # Compute pairwise cosine similarity
        similarity_matrix = matrix_normalized @ matrix_normalized.T

        edge_count = 0
        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                sim = float(similarity_matrix[i, j])
                if sim >= threshold:
                    self.graph.add_edge(
                        paper_ids[i],
                        paper_ids[j],
                        type="similar",
                        weight=sim,
                    )
                    self.graph.add_edge(
                        paper_ids[j],
                        paper_ids[i],
                        type="similar",
                        weight=sim,
                    )
                    edge_count += 1

        console.print(f"[cyan]Graph:[/cyan] Built {edge_count} similarity edges (threshold={threshold})")
        return edge_count

    def build_author_edges(self):
        """Create edges between papers that share authors."""
        from collections import defaultdict

        author_papers: dict[str, list[str]] = defaultdict(list)
        for paper_id, paper in self.papers.items():
            for author in paper.authors:
                author_papers[author.name.lower()].append(paper_id)

        edge_count = 0
        for author_name, pids in author_papers.items():
            if len(pids) > 1:
                for i in range(len(pids)):
                    for j in range(i + 1, len(pids)):
                        if not self.graph.has_edge(pids[i], pids[j]):
                            self.graph.add_edge(
                                pids[i],
                                pids[j],
                                type="same_author",
                                weight=0.5,
                                author=author_name,
                            )
                            edge_count += 1

        console.print(f"[cyan]Graph:[/cyan] Built {edge_count} same-author edges")
        return edge_count

    def add_topic_nodes(self, paper_topics: dict[str, list[str]]):
        """
        Add topic nodes and connect them to papers.

        Args:
            paper_topics: Dict mapping paper_id -> list of topic strings
        """
        topic_count = 0
        for paper_id, topics in paper_topics.items():
            if paper_id not in self.papers:
                continue
            for topic in topics:
                topic_node_id = f"topic:{topic.lower()}"
                if topic_node_id not in self.graph:
                    self.graph.add_node(
                        topic_node_id,
                        type="topic",
                        name=topic,
                    )
                    topic_count += 1
                self.graph.add_edge(
                    paper_id,
                    topic_node_id,
                    type="has_topic",
                    weight=1.0,
                )

        console.print(f"[cyan]Graph:[/cyan] Added {topic_count} topic nodes")

    def get_paper_node(self, paper_id: str) -> Optional[dict]:
        """Get paper node data."""
        if paper_id in self.graph:
            return dict(self.graph.nodes[paper_id])
        return None

    def get_paper_neighbors(self, paper_id: str, edge_type: Optional[str] = None) -> list[str]:
        """Get neighbors of a paper, optionally filtered by edge type."""
        if paper_id not in self.graph:
            return []

        neighbors = []
        for _, target, data in self.graph.edges(paper_id, data=True):
            if edge_type is None or data.get("type") == edge_type:
                neighbors.append(target)
        return neighbors

    def get_undirected_copy(self) -> nx.Graph:
        """Return an undirected copy of the graph (for community detection)."""
        return self.graph.to_undirected()

    def get_stats(self) -> dict:
        """Get graph statistics."""
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "paper"]
        topic_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "topic"]

        citation_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "cites"]
        sim_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "similar"]

        return {
            "total_nodes": self.num_nodes,
            "paper_nodes": len(paper_nodes),
            "topic_nodes": len(topic_nodes),
            "total_edges": self.num_edges,
            "citation_edges": len(citation_edges),
            "similarity_edges": len(sim_edges),
        }
