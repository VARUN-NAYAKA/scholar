"""
ScholAR - Search History Backend.
Stores search history and results in a local SQLite database.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from core.config import DATA_DIR

DB_PATH = DATA_DIR / "scholar_history.db"


def _get_conn():
    """Get a connection to the history database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            topic TEXT NOT NULL,
            papers_count INTEGER DEFAULT 0,
            clusters_count INTEGER DEFAULT 0,
            gaps_count INTEGER DEFAULT 0,
            trends_count INTEGER DEFAULT 0,
            coverage_score REAL DEFAULT 0,
            top_papers TEXT DEFAULT '[]',
            summary TEXT DEFAULT '',
            sections TEXT DEFAULT '[]'
        )
    """)
    conn.commit()
    return conn


def save_search(topic: str, result: dict) -> int:
    """Save a search result to history. Returns the record ID."""
    conn = _get_conn()

    report = result.get("report")
    analysis = result.get("analysis")
    papers = result.get("scored_papers", [])

    # Extract sections data
    sections_data = []
    if report:
        for sec in report.sections:
            sections_data.append({
                "title": sec.title,
                "content": sec.content[:2000] if sec.content else "",
                "order": sec.order,
            })

    # Extract top papers
    top_papers_data = []
    if report and report.top_recommendations:
        for rec in report.top_recommendations:
            top_papers_data.append({
                "title": rec.paper.title,
                "year": rec.paper.year,
                "citations": rec.paper.citation_count,
                "confidence": rec.confidence,
                "reason": rec.reason,
            })

    # Get executive summary
    summary = ""
    if report:
        for sec in report.sections:
            if sec.title == "Executive Summary":
                summary = sec.content[:1000] if sec.content else ""
                break

    cursor = conn.execute("""
        INSERT INTO search_history (timestamp, topic, papers_count, clusters_count,
            gaps_count, trends_count, coverage_score, top_papers, summary, sections)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        topic,
        len(papers),
        len(analysis.clusters) if analysis else 0,
        len(analysis.gaps) if analysis else 0,
        len(analysis.trends) if analysis else 0,
        analysis.coverage_score if analysis else 0,
        json.dumps(top_papers_data),
        summary,
        json.dumps(sections_data),
    ))

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_history(limit: int = 20) -> list[dict]:
    """Get recent search history."""
    conn = _get_conn()
    cursor = conn.execute("""
        SELECT id, timestamp, topic, papers_count, clusters_count,
               gaps_count, trends_count, coverage_score, summary
        FROM search_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "topic": row[2],
            "papers_count": row[3],
            "clusters_count": row[4],
            "gaps_count": row[5],
            "trends_count": row[6],
            "coverage_score": row[7],
            "summary": row[8],
        }
        for row in rows
    ]


def get_search_detail(record_id: int) -> dict:
    """Get full detail of a past search."""
    conn = _get_conn()
    cursor = conn.execute("""
        SELECT * FROM search_history WHERE id = ?
    """, (record_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    return {
        "id": row[0],
        "timestamp": row[1],
        "topic": row[2],
        "papers_count": row[3],
        "clusters_count": row[4],
        "gaps_count": row[5],
        "trends_count": row[6],
        "coverage_score": row[7],
        "top_papers": json.loads(row[8]),
        "summary": row[9],
        "sections": json.loads(row[10]),
    }
