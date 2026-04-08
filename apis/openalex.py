"""
ScholAR - OpenAlex API wrapper.
Docs: https://docs.openalex.org/
Free, no auth required. Very generous rate limits.
"""

import time
import requests
from typing import Optional
from core.config import OPENALEX_RPS, CROSSREF_MAILTO
from core.models import Paper, Author
from rich.console import Console

console = Console()

BASE_URL = "https://api.openalex.org"


def _rate_limit():
    time.sleep(1.0 / OPENALEX_RPS)


def search_papers(
    query: str,
    limit: int = 50,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
) -> list[Paper]:
    """
    Search OpenAlex for papers (called 'works').

    Args:
        query: Search string
        limit: Max results
        from_year: Filter papers from this year
        to_year: Filter papers up to this year

    Returns:
        List of Paper objects
    """
    papers = []
    page = 1
    per_page = min(limit, 50)

    while len(papers) < limit:
        params = {
            "search": query,
            "per_page": per_page,
            "page": page,
            "sort": "relevance_score:desc",
        }

        # Use mailto for polite pool (faster)
        if CROSSREF_MAILTO:
            params["mailto"] = CROSSREF_MAILTO

        # Year filter
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        _rate_limit()

        try:
            resp = requests.get(f"{BASE_URL}/works", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            console.print(f"[red]OpenAlex API error: {e}[/red]")
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            paper = _parse_work(item)
            if paper:
                papers.append(paper)

        # Check if more pages exist
        meta = data.get("meta", {})
        total_count = meta.get("count", 0)
        if page * per_page >= total_count:
            break

        page += 1

    console.print(f"[green]OpenAlex:[/green] Found {len(papers)} papers for '{query}'")
    return papers[:limit]


def _parse_work(data: dict) -> Optional[Paper]:
    """Parse an OpenAlex work into a Paper object."""
    try:
        openalex_id = data.get("id", "")
        # Use OpenAlex ID as paper_id
        paper_id = f"openalex:{openalex_id.split('/')[-1]}" if openalex_id else None
        if not paper_id:
            return None

        title = data.get("title", "")
        if not title:
            return None

        # Extract abstract from inverted index
        abstract = ""
        abstract_index = data.get("abstract_inverted_index")
        if abstract_index:
            abstract = _reconstruct_abstract(abstract_index)

        # Authors
        authors = []
        for authorship in data.get("authorships", []) or []:
            author_data = authorship.get("author", {})
            institution = ""
            institutions = authorship.get("institutions", [])
            if institutions:
                institution = institutions[0].get("display_name", "")

            authors.append(
                Author(
                    author_id=f"openalex:{author_data.get('id', '').split('/')[-1]}",
                    name=author_data.get("display_name", "Unknown"),
                    affiliation=institution or None,
                )
            )

        # DOI
        doi = data.get("doi", "")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        # Venue / Source
        venue = ""
        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name", "")

        # Referenced works (as OpenAlex IDs)
        refs = []
        for ref_url in data.get("referenced_works", []) or []:
            ref_id = f"openalex:{ref_url.split('/')[-1]}"
            refs.append(ref_id)

        # Concepts/Topics
        topics = []
        for concept in data.get("concepts", []) or []:
            if concept.get("score", 0) > 0.3:
                topics.append(concept.get("display_name", ""))

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=data.get("publication_year"),
            doi=doi or None,
            url=data.get("id", ""),
            venue=venue,
            citation_count=data.get("cited_by_count", 0) or 0,
            reference_count=len(refs),
            references=refs[:50],  # Limit to avoid huge lists
            source_api="openalex",
            topics=topics,
            fields_of_study=[t.get("display_name", "") for t in (data.get("topics") or [])[:5]],
        )
    except Exception as e:
        console.print(f"[yellow]OpenAlex parse error: {e}[/yellow]")
        return None


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    # Create list of (position, word) tuples
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Sort by position and join
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)
