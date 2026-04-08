"""
ScholAR - CrossRef API wrapper.
Docs: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
"""

import time
import requests
from typing import Optional
from core.config import CROSSREF_RPS, CROSSREF_MAILTO
from core.models import Paper, Author
from rich.console import Console

console = Console()

BASE_URL = "https://api.crossref.org/works"


def _rate_limit():
    time.sleep(1.0 / CROSSREF_RPS)


def search_papers(
    query: str,
    limit: int = 50,
    from_year: Optional[int] = None,
) -> list[Paper]:
    """
    Search CrossRef for papers.

    Args:
        query: Search string
        limit: Max results
        from_year: Filter papers from this year onward

    Returns:
        List of Paper objects
    """
    papers = []
    offset = 0
    per_page = min(limit, 50)

    headers = {}
    if CROSSREF_MAILTO:
        headers["User-Agent"] = f"ScholAR/1.0 (mailto:{CROSSREF_MAILTO})"

    while len(papers) < limit:
        params = {
            "query": query,
            "rows": per_page,
            "offset": offset,
            "sort": "relevance",
            "order": "desc",
            "select": "DOI,title,abstract,author,published-print,published-online,"
                      "is-referenced-by-count,references-count,container-title,URL,subject",
        }

        if from_year:
            params["filter"] = f"from-pub-date:{from_year}"

        _rate_limit()

        try:
            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            console.print(f"[red]CrossRef API error: {e}[/red]")
            break

        items = data.get("message", {}).get("items", [])
        if not items:
            break

        for item in items:
            paper = _parse_item(item)
            if paper:
                papers.append(paper)

        total = data.get("message", {}).get("total-results", 0)
        offset += per_page
        if offset >= total:
            break

    console.print(f"[green]CrossRef:[/green] Found {len(papers)} papers for '{query}'")
    return papers[:limit]


def _parse_item(data: dict) -> Optional[Paper]:
    """Parse a CrossRef work item into a Paper object."""
    try:
        doi = data.get("DOI", "")
        if not doi:
            return None

        paper_id = f"crossref:{doi}"

        # Title
        titles = data.get("title", [])
        title = titles[0] if titles else ""
        if not title:
            return None

        # Abstract (CrossRef abstracts often contain XML tags)
        abstract = data.get("abstract", "")
        if abstract:
            # Strip basic XML/HTML tags
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)

        # Authors
        authors = []
        for a in data.get("author", []) or []:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(
                    Author(
                        author_id=f"crossref:{name.lower().replace(' ', '_')}",
                        name=name,
                        affiliation=(
                            a.get("affiliation", [{}])[0].get("name")
                            if a.get("affiliation")
                            else None
                        ),
                    )
                )

        # Year (prefer print date, fall back to online)
        year = None
        for date_field in ["published-print", "published-online"]:
            date_parts = data.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break

        # Venue
        container = data.get("container-title", [])
        venue = container[0] if container else ""

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            doi=doi,
            url=data.get("URL", ""),
            venue=venue,
            citation_count=data.get("is-referenced-by-count", 0) or 0,
            reference_count=data.get("references-count", 0) or 0,
            source_api="crossref",
            fields_of_study=data.get("subject", []) or [],
        )
    except Exception as e:
        console.print(f"[yellow]CrossRef parse error: {e}[/yellow]")
        return None
