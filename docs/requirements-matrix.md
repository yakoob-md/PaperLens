# PaperLens — 10 Minimum Requirements Code & Architecture Map

This document maps all 10 evaluation criteria to exact files, functions, and architecture components for professors and reviewers.

| # | Requirement | Implementation | Where to verify |
|---|---|---|---|
| 1 | Shared context | Review state passes literature to CrewAI, specialist notes to synthesis, and findings to the critic. | Context & State Updates |
| 2 | MCP tools | MCP SDK initializes a stdio session with FastMCP and invokes both text statistics and literature search. | MCP Tools & Protocol |
| 3 | Pydantic output | FastAPI returns a validated `ReviewReport` from both review endpoints. | JSON export |
| 4 | Agentic design patterns | The orchestrator delegates to a sequential specialist crew and an independent critic. | Agentic Activity |
| 5 | Callbacks | CrewAI task and ADK model callbacks capture intermediate outputs. Tool and model failures are recorded. | Critique & logs |
| 6 | A2A | The A2A SDK resolves the agent card and sends `message/send` to the ADK critic. | A2A Inter-Agent Protocol |
| 7 | CrewAI and ADK | CrewAI runs the three specialists and Google ADK runs the critic. Gemini and Groq provide inference. | Agentic Activity and A2A |
| 8 | Guardrails | Input checks run before inference. MCP payloads are validated before use. Findings are grounded against retrieved IDs and final scores are schema-validated. | Guardrails & Security |
| 9 | Observability | Local trace events, model usage and provenance are included in each report. Optional Langfuse export sends spans, statuses, and generation usage. | Critique & logs and [Langfuse setup](../README.md#langfuse-tracing) |
| 10 | Evaluation | The API benchmark evaluates each case, reports measured latency and estimated provider charges, and checks required stages. | Evaluation & Cost and `backend/evaluation/run_eval.py` |

---

## 💡 New Feature: Research Idea Novelty Checker ("Check Research Idea")

| Feature Dimension | Implementation Location | Capability Description |
|---|---|---|
| **Multi-Format Input** | `backend/app/idea_checker.py:extract_text_from_file` | Accepts raw text descriptions OR uploaded drafts in **PDF** and **DOCX** formats with automated text extraction. |
| **Concept & Domain Analysis** | `backend/app/idea_checker.py:analyze_research_idea` | Deconstructs idea into: problem statement, objectives, proposed methodology, key concepts/technologies, domain, and keywords. |
| **Verifiable Academic Search** | `backend/app/idea_checker.py:search_similar_papers` | Searches authentic literature across arXiv, OpenAlex, and CrossRef without fabricating DOIs, authors, or publication URLs. |
| **Novelty Categorization & Scoring** | `backend/app/idea_checker.py:assess_novelty` | Evaluates novelty into 4 tiers (`Highly Explored`, `Partially Explored`, `Limited Existing Research`, `Potentially Novel`) + 0–100 indicative score and academic disclaimer. |
| **5-Section Results Dashboard** | `frontend/src/App.jsx` (`appMode === "idea"`) | 1. Idea Overview<br>2. Novelty Assessment<br>3. Similar Research Papers (with direct clickable links)<br>4. Research Gap Identification<br>5. Actionable Recommendations. |

