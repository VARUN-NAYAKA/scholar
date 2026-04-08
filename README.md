# 🎓 ScholAR — Autonomous Research Intelligence Agent

> From research topic to complete literature review in minutes.

ScholAR is an **autonomous AI agent** that takes a research topic, author name, or uploaded PDF as input and automatically:
- 🔍 **Searches** 4 academic APIs (Semantic Scholar, ArXiv, OpenAlex, CrossRef)
- 👤 **Finds** all papers by a specific researcher
- 📄 **Analyzes** uploaded PDFs to seed related research discovery
- 🕸️ **Builds** an interactive knowledge graph of related research
- 📊 **Analyzes** clusters, trends, gaps, and contradictions
- 📝 **Generates** a complete literature review report
- 🏆 **Recommends** the top 3 most relevant papers

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/VARUN-NAYAKA/scholar.git
cd scholar

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the app
streamlit run app.py
```

## 🏗️ Architecture

```
ScholAR
├── app.py                  # Streamlit dashboard
├── agents/
│   ├── orchestrator.py     # LangGraph state machine (the brain)
│   ├── search_agent.py     # Multi-API search + scoring
│   ├── analysis_agent.py   # Gap/trend/cluster analysis
│   └── report_agent.py     # Literature review generator + PDF
├── apis/
│   ├── semantic_scholar.py
│   ├── arxiv_client.py
│   ├── openalex.py
│   └── crossref.py
├── core/
│   ├── config.py           # Settings + environment
│   ├── models.py           # Pydantic data models
│   ├── prompts.py          # LLM prompt templates
│   ├── history.py          # Search history backend
│   └── pdf_utils.py        # PDF text extraction + seed topic
└── graph/
    ├── builder.py           # NetworkX knowledge graph
    ├── algorithms.py        # Louvain, PageRank, centrality
    └── visualizer.py        # PyVis interactive visualization
```

## 🧠 How It Works

1. **Query Expansion** — Gemini LLM expands your topic into 12+ diverse search queries
2. **Parallel Search** — All 4 academic APIs are searched simultaneously
3. **Deduplication** — Papers are deduplicated via DOI + title hashing
4. **Relevance Scoring** — Weighted formula: semantic similarity + citations + recency + venue prestige
5. **Knowledge Graph** — NetworkX graph with citation, similarity, author, and topic edges
6. **Community Detection** — Louvain algorithm identifies research clusters
7. **Gap & Trend Analysis** — LLM + algorithmic analysis finds under-explored areas and temporal trends
8. **Report Generation** — Structured literature review with APA citations
9. **Autonomous Loop** — Agent decides whether to search more or finalize based on coverage score

## 🔑 API Keys Required

| Key | Required | Source |
|-----|----------|--------|
| `GEMINI_API_KEY` | ✅ Yes | [Google AI Studio](https://aistudio.google.com/apikey) |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | [Semantic Scholar](https://www.semanticscholar.org/product/api) |
| `CROSSREF_MAILTO` | Optional | Your email for polite pool access |

## 📸 Features

- **3 Research Modes**: Topic Search, Author Search, PDF Upload
- **Premium Dark UI** with glassmorphism design
- **Interactive Knowledge Graph** (PyVis)
- **Publication Trend Charts** (Plotly)
- **Research Gap Detection** with confidence scores
- **Top 3 Paper Recommendations** with reasoning
- **PDF Report Export**
- **Search History** persistence
- **PDF Paper Analysis** — upload any paper to discover related work

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.0 Flash
- **Agent Framework**: LangGraph
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Graph**: NetworkX + python-louvain
- **Frontend**: Streamlit
- **Visualization**: Plotly + PyVis

---

*Built with ❤️ for the Agentic AI Hackathon*
