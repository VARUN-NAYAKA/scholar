"""
ScholAR - Semantic Scholar API wrapper.
Docs: https://api.semanticscholar.org/api-docs/graph
"""

import time
import requests
from typing import Optional
from core.config import SEMANTIC_SCHOLAR_API_KEY, SEMANTIC_SCHOLAR_RPS
from core.models import Paper, Author
from rich.console import Console

console = Console()

BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_URL = f"{BASE_URL}/paper/search"
PAPER_URL = f"{BASE_URL}/paper"

HEADERS = {}
if SEMANTIC_SCHOLAR_API_KEY:
    HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY


def _rate_limit():
    """Simple rate limiter."""
    time.sleep(1.0 / SEMANTIC_SCHOLAR_RPS)


def search_papers(
    query: str,
    limit: int = 50,
    year_range: Optional[str] = None,
    fields_of_study: Optional[list[str]] = None,
) -> list[Paper]:
    """
    Search for papers on Semantic Scholar.

    Args:
        query: Search query string
        limit: Maximum number of results (max 100)
        year_range: e.g., "2020-2024" or "2020-"
        fields_of_study: e.g., ["Computer Science", "Medicine"]

    Returns:
        List of Paper objects
    """
    papers = []
    offset = 0
    per_page = min(limit, 100)

    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,tldr,fieldsOfStudy,"
        "citations.paperId,references.paperId"
    )

    while len(papers) < limit:
        params = {
            "query": query,
            "offset": offset,
            "limit": per_page,
            "fields": fields,
        }

        if year_range:
            params["year"] = year_range
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        _rate_limit()

        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            console.print(f"[red]Semantic Scholar API error: {e}[/red]")
            break

        results = data.get("data", [])
        if not results:
            break

        for item in results:
            paper = _parse_paper(item)
            if paper:
                papers.append(paper)

        total_available = data.get("total", 0)
        offset += per_page

        if offset >= total_available:
            break

    console.print(f"[green]Semantic Scholar:[/green] Found {len(papers)} papers for '{query}'")
    return papers[:limit]


def get_paper_details(paper_id: str) -> Optional[Paper]:
    """Fetch detailed info for a specific paper by its Semantic Scholar ID."""
    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,tldr,fieldsOfStudy,"
        "citations.paperId,references.paperId"
    )

    _rate_limit()

    try:
        resp = requests.get(
            f"{PAPER_URL}/{paper_id}",
            params={"fields": fields},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return _parse_paper(resp.json())
    except requests.RequestException as e:
        console.print(f"[red]Semantic Scholar detail error: {e}[/red]")
        return None


def _parse_paper(data: dict) -> Optional[Paper]:
    """Parse API response into a Paper object."""
    if not data.get("paperId") or not data.get("title"):
        return None

    authors = []
    for a in data.get("authors", []) or []:
        authors.append(
            Author(
                author_id=a.get("authorId", ""),
                name=a.get("name", "Unknown"),
            )
        )

    # Extract DOI from externalIds
    external_ids = data.get("externalIds") or {}
    doi = external_ids.get("DOI", None)

    # Extract references and citations as ID lists
    refs = [r["paperId"] for r in (data.get("references") or []) if r.get("paperId")]
    cites = [c["paperId"] for c in (data.get("citations") or []) if c.get("paperId")]

    # TLDR
    tldr_data = data.get("tldr")
    tldr = tldr_data.get("text", "") if isinstance(tldr_data, dict) else None

    return Paper(
        paper_id=data["paperId"],
        title=data["title"],
        abstract=data.get("abstract") or "",
        authors=authors,
        year=data.get("year"),
        doi=doi,
        url=data.get("url", ""),
        venue=data.get("venue", ""),
        citation_count=data.get("citationCount", 0) or 0,
        reference_count=data.get("referenceCount", 0) or 0,
        references=refs,
        citations=cites,
        source_api="semantic_scholar",
        tldr=tldr,
        fields_of_study=data.get("fieldsOfStudy") or [],
    )


def search_author_papers(author_name: str, limit: int = 50) -> list[Paper]:
    """
    Multi-source author search with fuzzy matching.
    Strategy:
      1. Semantic Scholar Author API
      2. OpenAlex Author API (very robust, handles name variations)
      3. Semantic Scholar paper search with author name
      4. OpenAlex paper search with author name
    Results are combined and deduplicated.
    """
    console.print(f"[magenta]Searching for author: '{author_name}'[/magenta]")

    all_papers = []

    # ── Strategy 1: Semantic Scholar Author API ──
    ss_papers = _ss_author_search(author_name, limit)
    if ss_papers:
        console.print(f"[green]SemanticScholar Author API:[/green] {len(ss_papers)} papers")
        all_papers.extend(ss_papers)

    # ── Strategy 2: OpenAlex Author API (best for fuzzy/uncommon names) ──
    oa_papers = _openalex_author_search(author_name, limit)
    if oa_papers:
        console.print(f"[green]OpenAlex Author API:[/green] {len(oa_papers)} papers")
        all_papers.extend(oa_papers)

    # ── Strategy 3: Semantic Scholar paper search with author name  ──
    if len(all_papers) < 5:
        console.print(f"[yellow]Trying SS paper search for '{author_name}'...[/yellow]")
        ss_keyword = search_papers(author_name, limit=limit)
        # Filter to papers that actually have this author (fuzzy)
        ss_keyword = _filter_by_author(ss_keyword, author_name)
        all_papers.extend(ss_keyword)

    # ── Strategy 4: OpenAlex paper search with author filter ──
    if len(all_papers) < 5:
        console.print(f"[yellow]Trying OpenAlex paper search for '{author_name}'...[/yellow]")
        try:
            from apis.openalex import search_papers as oa_search
            oa_keyword = oa_search(author_name, limit=limit)
            oa_keyword = _filter_by_author(oa_keyword, author_name)
            all_papers.extend(oa_keyword)
        except Exception as e:
            console.print(f"[red]OpenAlex paper search error: {e}[/red]")

    # Deduplicate by title (normalized)
    seen_titles = set()
    unique_papers = []
    for p in all_papers:
        norm_title = p.title.lower().strip()
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_papers.append(p)

    console.print(f"[green]Author '{author_name}':[/green] Total {len(unique_papers)} unique papers")
    return unique_papers[:limit]


def _ss_author_search(author_name: str, limit: int) -> list[Paper]:
    """Try Semantic Scholar Author Search API."""
    _rate_limit()
    try:
        resp = requests.get(
            f"{BASE_URL}/author/search",
            params={"query": author_name, "limit": 5},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]SS Author API error: {e}[/red]")
        return []

    authors_found = data.get("data", [])
    if not authors_found:
        return []

    # Pick best match using fuzzy name comparison
    best_author = _pick_best_author(authors_found, author_name)
    author_id = best_author.get("authorId")
    author_display = best_author.get("name", author_name)
    console.print(f"[green]SS Found:[/green] {author_display} (ID: {author_id})")

    if not author_id:
        return []

    _rate_limit()
    fields = (
        "paperId,title,abstract,authors,year,citationCount,"
        "referenceCount,venue,externalIds,url,fieldsOfStudy"
    )

    try:
        resp = requests.get(
            f"{BASE_URL}/author/{author_id}/papers",
            params={"fields": fields, "limit": min(limit, 100)},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]SS papers fetch error: {e}[/red]")
        return []

    papers = []
    for item in data.get("data", []):
        paper = _parse_paper(item)
        if paper:
            papers.append(paper)

    return papers[:limit]


def _openalex_author_search(author_name: str, limit: int) -> list[Paper]:
    """Search OpenAlex for an author and return their works. Very robust for
    fuzzy/uncommon names since OpenAlex indexes 250M+ works."""
    try:
        resp = requests.get(
            "https://api.openalex.org/authors",
            params={"search": author_name, "per_page": 5},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]OpenAlex Author API error: {e}[/red]")
        return []

    authors = data.get("results", [])
    if not authors:
        return []

    # Pick best match using fuzzy name comparison
    best = _pick_best_openalex_author(authors, author_name)
    oa_id = best.get("id", "")
    display_name = best.get("display_name", author_name)
    console.print(f"[green]OpenAlex Found:[/green] {display_name} (ID: {oa_id})")

    if not oa_id:
        return []

    # Fetch works by this author
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "filter": f"authorships.author.id:{oa_id}",
                "per_page": min(limit, 100),
                "sort": "cited_by_count:desc",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        console.print(f"[red]OpenAlex works fetch error: {e}[/red]")
        return []

    from apis.openalex import _parse_work
    papers = []
    for item in data.get("results", []):
        paper = _parse_work(item)
        if paper:
            papers.append(paper)

    return papers[:limit]


def _pick_best_author(authors: list, query_name: str) -> dict:
    """Pick the best matching author from Semantic Scholar results using fuzzy matching."""
    query_lower = query_name.lower().strip()
    query_parts = set(query_lower.split())

    best = authors[0]
    best_score = 0

    for author in authors:
        name = (author.get("name") or "").lower()
        name_parts = set(name.split())

        # Count matching name parts
        overlap = len(query_parts & name_parts)
        total = max(len(query_parts), 1)
        score = overlap / total

        # Bonus for exact match
        if name == query_lower:
            score = 2.0
        # Bonus for substring match
        elif query_lower in name or name in query_lower:
            score += 0.5

        if score > best_score:
            best_score = score
            best = author

    return best


def _pick_best_openalex_author(authors: list, query_name: str) -> dict:
    """Pick the best matching author from OpenAlex results using fuzzy matching."""
    query_lower = query_name.lower().strip()
    query_parts = set(query_lower.split())

    best = authors[0]
    best_score = 0

    for author in authors:
        name = (author.get("display_name") or "").lower()
        name_parts = set(name.split())

        overlap = len(query_parts & name_parts)
        total = max(len(query_parts), 1)
        score = overlap / total

        # Bonus for exact match
        if name == query_lower:
            score = 2.0
        elif query_lower in name or name in query_lower:
            score += 0.5

        # Bonus for higher works count (more likely to be the right person)
        works_count = author.get("works_count", 0)
        if works_count > 50:
            score += 0.3
        elif works_count > 10:
            score += 0.1

        if score > best_score:
            best_score = score
            best = author

    return best


def _filter_by_author(papers: list[Paper], author_name: str) -> list[Paper]:
    """Filter papers to only include those with a matching author (fuzzy)."""
    query_parts = set(author_name.lower().split())
    filtered = []

    for paper in papers:
        for author in paper.authors:
            name_parts = set(author.name.lower().split())
            overlap = len(query_parts & name_parts)
            # If at least half the name parts match, keep it
            if overlap >= max(len(query_parts) // 2, 1):
                filtered.append(paper)
                break

    return filtered
