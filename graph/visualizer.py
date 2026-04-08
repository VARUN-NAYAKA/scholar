"""
ScholAR - Interactive Knowledge Graph Visualization using PyVis.
"""

from pyvis.network import Network
from pathlib import Path
from core.config import DATA_DIR
from rich.console import Console

console = Console()

# Color palette for communities
COMMUNITY_COLORS = [
    "#e94560", "#0f3460", "#533483", "#16213e",
    "#f39c12", "#2ecc71", "#3498db", "#e74c3c",
    "#9b59b6", "#1abc9c", "#e67e22", "#2980b9",
    "#c0392b", "#27ae60", "#8e44ad", "#d35400",
]


def create_interactive_graph(
    graph,
    partition: dict[str, int] = None,
    pagerank: dict[str, float] = None,
    paper_data: dict = None,
    output_filename: str = "knowledge_graph.html",
    height: str = "800px",
    width: str = "100%",
) -> str:
    """
    Create an interactive HTML visualization of the knowledge graph.
    """
    net = Network(
        height=height,
        width=width,
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Physics configuration for nice layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            }
        }
    }
    """)

    # Normalize PageRank for node sizing
    if pagerank:
        max_pr = max(pagerank.values()) if pagerank.values() else 1
    else:
        max_pr = 1

    # Add nodes
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("type", "paper")

        if node_type == "topic":
            net.add_node(
                node_id,
                label=data.get("name", node_id),
                color="#f39c12",
                shape="diamond",
                size=15,
                title=f"Topic: {data.get('name', '')}",
            )
        else:
            # Paper nodes
            comm_id = partition.get(node_id, 0) if partition else 0
            color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]

            pr_score = pagerank.get(node_id, 0) if pagerank else 0
            size = 10 + (pr_score / max_pr) * 30 if max_pr > 0 else 10

            # Build tooltip using PLAIN TEXT (no HTML tags - PyVis escapes them)
            paper = paper_data.get(node_id) if paper_data else None
            if paper:
                title_text = paper.title
                authors_str = ", ".join(a.name for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += " et al."
                tooltip = (
                    f"{paper.title}\n"
                    f"──────────────────────\n"
                    f"Year: {paper.year or 'N/A'}\n"
                    f"Citations: {paper.citation_count}\n"
                    f"Venue: {paper.venue or 'N/A'}\n"
                    f"Authors: {authors_str}\n"
                    f"Cluster: {comm_id}\n"
                    f"Relevance: {paper.relevance_score:.3f}"
                )
            else:
                title_text = data.get("title", node_id)[:40]
                tooltip = data.get("title", node_id)

            net.add_node(
                node_id,
                label=title_text[:40] + "..." if len(title_text) > 40 else title_text,
                color=color,
                size=size,
                title=tooltip,
                shape="dot",
            )

    # Add edges
    for source, target, data in graph.edges(data=True):
        edge_type = data.get("type", "cites")

        edge_config = {
            "cites": {"color": "#555555", "width": 1, "dashes": False},
            "similar": {"color": "#3498db", "width": 2, "dashes": True},
            "same_author": {"color": "#2ecc71", "width": 1, "dashes": [5, 5]},
            "has_topic": {"color": "#f39c12", "width": 1, "dashes": [2, 2]},
        }

        config = edge_config.get(edge_type, edge_config["cites"])

        net.add_edge(
            source,
            target,
            color=config["color"],
            width=config["width"],
            dashes=config["dashes"],
            title=f"{edge_type} (weight: {data.get('weight', 1.0):.2f})",
        )

    # Save to file
    output_path = DATA_DIR / output_filename
    net.save_graph(str(output_path))

    console.print(f"[green]Visualization:[/green] Saved interactive graph to {output_path}")
    return str(output_path)
