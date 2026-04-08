"""
ScholAR - All LLM prompt templates used by agents.
"""

# ──────────────────────────────────────────────
# QUERY EXPANSION PROMPT
# ──────────────────────────────────────────────
QUERY_EXPANSION_PROMPT = """You are a research assistant helping to find academic papers on a specific topic.

Given the research topic below, generate {num_queries} diverse search queries that would help find relevant papers across different aspects of this topic.

**Research Topic:** {topic}

Generate queries in these categories:
1. **Direct** (2-3 queries): Exact topic and close variations
2. **Synonyms** (2-3 queries): Alternative terminology for the same concepts
3. **Broader** (1-2 queries): Wider field that includes this topic
4. **Narrower** (2-3 queries): Specific sub-topics or applications
5. **Related** (2-3 queries): Connected concepts that intersect with this topic

Return ONLY a JSON array of objects, each with "query" and "strategy" keys:
[
    {{"query": "example search query", "strategy": "direct"}},
    ...
]

Do NOT include any text outside the JSON array."""


# ──────────────────────────────────────────────
# TOPIC EXTRACTION PROMPT
# ──────────────────────────────────────────────
TOPIC_EXTRACTION_PROMPT = """Analyze the following paper title and abstract. Extract the main research topics and keywords.

**Title:** {title}
**Abstract:** {abstract}

Return ONLY a JSON object:
{{
    "topics": ["topic1", "topic2", "topic3"],
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

Keep topics broad (e.g., "natural language processing", "attention mechanisms") and keywords specific (e.g., "multi-head attention", "BERT fine-tuning").
Do NOT include any text outside the JSON."""


# ──────────────────────────────────────────────
# GAP DETECTION PROMPT
# ──────────────────────────────────────────────
GAP_DETECTION_PROMPT = """You are a research analyst. Based on the following research clusters and their papers, identify potential research gaps — areas that are under-explored or missing.

**Research Topic:** {topic}

**Identified Clusters:**
{clusters_text}

**Key Statistics:**
- Total papers analyzed: {total_papers}
- Number of clusters: {num_clusters}
- Time range: {year_range}

Identify 3-5 research gaps. For each gap, explain:
1. What is missing or under-explored
2. Why it matters
3. Suggested research directions

Return ONLY a JSON array:
[
    {{
        "description": "Description of the gap",
        "confidence": 0.8,
        "related_topics": ["topic1", "topic2"],
        "suggested_directions": ["direction1", "direction2"]
    }}
]

Do NOT include any text outside the JSON array."""


# ──────────────────────────────────────────────
# CONTRADICTION DETECTION PROMPT
# ──────────────────────────────────────────────
CONTRADICTION_DETECTION_PROMPT = """Analyze the following two paper abstracts and determine if they present contradictory findings or conclusions.

**Paper A:**
Title: {title_a}
Abstract: {abstract_a}

**Paper B:**
Title: {title_b}
Abstract: {abstract_b}

Return ONLY a JSON object:
{{
    "is_contradictory": true/false,
    "paper_a_claim": "Main claim from Paper A",
    "paper_b_claim": "Main claim from Paper B", 
    "description": "Brief description of the contradiction (or why they agree)",
    "confidence": 0.0-1.0
}}

Do NOT include any text outside the JSON."""


# ──────────────────────────────────────────────
# REPORT SECTION WRITER PROMPT
# ──────────────────────────────────────────────
REPORT_SECTION_PROMPT = """You are writing a section of an academic literature review.

**Overall Topic:** {topic}
**Section Title:** {section_title}
**Section Purpose:** {section_purpose}

**Relevant Papers:**
{papers_text}

**Analysis Insights:**
{insights_text}

Write a comprehensive, well-structured section for a literature review. Requirements:
- Use academic writing style
- Cite papers using [Author, Year] format
- Synthesize information across papers, don't just summarize each one
- Highlight agreements, disagreements, and trends
- Be specific about methods and findings
- Length: {target_length} words

Write the section content directly. Do NOT include the section title."""


# ──────────────────────────────────────────────
# EXECUTIVE SUMMARY PROMPT
# ──────────────────────────────────────────────
EXECUTIVE_SUMMARY_PROMPT = """Write an executive summary for a literature review on the following topic.

**Topic:** {topic}
**Key Statistics:**
- Papers analyzed: {total_papers}
- Research clusters identified: {num_clusters}
- Time span: {year_range}
- Research gaps found: {num_gaps}
- Key contradictions: {num_contradictions}

**Cluster Summaries:**
{cluster_summaries}

**Key Findings:**
{key_findings}

Write a concise executive summary (200-300 words) that:
1. Introduces the topic and scope of the review
2. Highlights the main research themes
3. Summarizes key findings and trends
4. Mentions notable gaps and contradictions
5. Provides recommendations for future research

Use academic writing style."""


# ──────────────────────────────────────────────
# TREND ANALYSIS PROMPT
# ──────────────────────────────────────────────
TREND_ANALYSIS_PROMPT = """Analyze the following publication data over time for a research topic and identify trends.

**Research Topic:** {topic}

**Papers per Year:**
{yearly_data}

**Topic Distribution Over Time:**
{topic_yearly_data}

Identify 3-5 notable trends. For each trend, provide:
1. A description of the trend
2. The topic it relates to
3. Whether it's rising, declining, stable, or emerging
4. The approximate start year
5. How strong the trend is (0.0 to 1.0)

Return ONLY a JSON array:
[
    {{
        "description": "Description of the trend",
        "topic": "Related topic",
        "direction": "rising",
        "start_year": 2020,
        "strength": 0.8
    }}
]

Do NOT include any text outside the JSON array."""


# ──────────────────────────────────────────────
# STOPPING DECISION PROMPT
# ──────────────────────────────────────────────
STOPPING_DECISION_PROMPT = """You are the planning module of a research agent. Based on the current state, decide whether to search for more papers or proceed to report generation.

**Current State:**
- Topic: {topic}
- Iteration: {iteration}/{max_iterations}
- Total papers collected: {total_papers}
- Number of clusters: {num_clusters}
- Coverage score: {coverage_score:.2f}
- New papers found last round: {new_papers_last}
- Previously searched queries: {searched_queries}

**Decision Criteria:**
- Minimum papers needed: {min_papers}
- Minimum clusters needed: {min_clusters}

If you decide to search more, suggest 3-5 NEW queries targeting gaps.

Return ONLY a JSON object:
{{
    "decision": "continue" or "stop",
    "reason": "Explanation of your decision",
    "new_queries": ["query1", "query2"] 
}}

Only include new_queries if decision is "continue".
Do NOT include any text outside the JSON."""
