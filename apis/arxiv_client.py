"""
ScholAR - ArXiv API wrapper.
Docs: https://info.arxiv.org/help/api/index.html
"""

import time
import arxiv
from typing import Optional
from core.config import ARXIV_RPS
from core.models import Paper, Author
from rich.console import Console

console = Console()


def _rate_limit():
    time.sleep(1.0 / ARXIV_RPS)


def search_papers(
    query: str,
    limit: int = 50,
    sort_by: str = "relevance",
) -> list[Paper]:
    """
    Search ArXiv for papers.

    Args:
        query: Search query string
        limit: Maximum results
        sort_by: "relevance" or "submittedDate"

    Returns:
        List of Paper objects
    """
    sort_criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client(
        page_size=min(limit, 100),
        delay_seconds=1.0 / ARXIV_RPS,
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=sort_criterion,
    )

    papers = []
    try:
        for result in client.results(search):
            paper = _parse_result(result)
            if paper:
                papers.append(paper)
    except Exception as e:
        console.print(f"[red]ArXiv API error: {e}[/red]")

    console.print(f"[green]ArXiv:[/green] Found {len(papers)} papers for '{query}'")
    return papers


def _parse_result(result) -> Optional[Paper]:
    """Parse an arxiv.Result into a Paper object."""
    try:
        # Generate a consistent ID from arxiv ID
        arxiv_id = result.entry_id.split("/abs/")[-1]
        paper_id = f"arxiv:{arxiv_id}"

        authors = [
            Author(
                author_id=f"arxiv_author:{a.name}",
                name=a.name,
            )
            for a in result.authors
        ]

        year = result.published.year if result.published else None

        # Extract DOI if available
        doi = result.doi if hasattr(result, "doi") else None

        return Paper(
            paper_id=paper_id,
            title=result.title,
            abstract=result.summary or "",
            authors=authors,
            year=year,
            doi=doi,
            url=result.entry_id,
            venue="arXiv",
            citation_count=0,  # ArXiv doesn't provide citation counts
            source_api="arxiv",
            fields_of_study=[cat.replace(".", " ") for cat in (result.categories or [])],
        )
    except Exception as e:
        console.print(f"[yellow]ArXiv parse error: {e}[/yellow]")
        return None
