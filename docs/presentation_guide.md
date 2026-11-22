# PaperLens — Presentation & Demonstration Guide

> **Project Goal**: An Agentic AI Research Paper Review System that goes beyond superficial summaries to rigorously stress-test scientific methodology, evaluate literature differentiation, enforce multi-layer guardrails, and foster multi-agent debate with challenge questions.

---

## 🚀 Quick Launch Instructions

### 1. Start the Backend API (FastAPI + FastMCP + A2A)
```powershell
cd c:\Users\HP\Downloads\PaperLens_Starter_Project\PaperLens\backend
..\..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
- API Health Check: `http://localhost:8000/health`
- A2A Agent Card: `http://localhost:8000/.well-known/agent-card.json`
- Interactive API Docs: `http://localhost:8000/docs`

### 2. Start the Frontend Dashboard (React + Vite + Tailwind)
```powershell
cd c:\Users\HP\Downloads\PaperLens_Starter_Project\PaperLens\frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🎯 How to Showcase the 5 Requested Presentation Features

### 1. Challenge Questions Thrown by Agents & Full Conversation
- **Where to click in UI**: Click the **"Agent Debate & Challenges"** tab (marked with ⚡).
- **What to show**:
  - Show the 4 debate rounds:
    - **Round 1**: Opening & Literature Retrieval
    - **Round 2**: Specialist Interrogation (Novelty Analyst, Methodology Reviewer, Evidence Reviewer)
    - **Round 3**: A2A Adversarial Cross-Examination (Google ADK Critic)
    - **Round 4**: Author Rebuttal Statement & Panel Consensus
  - Point out the highlighted **Challenge Question Callout Cards**:
    - **Challenge #1 (Architectural Novelty)**: Quizzing the author on semantic overlap with retrieved literature.
    - **Challenge #2 (Statistical Hygiene)**: Interrogating the single 80/20 train-test split and narrow 0.6% margin without random seeds.
    - **Challenge #3 (Component Ablation)**: Challenging the omission of an ablation study.
    - **Challenge #4 (Adversarial Stress-Test)**: The Google ADK critic demanding a Major Revision.
  - Use the top filter buttons: `All Turns`, `⚡ Challenge Questions Only`, `🛡️ A2A Adversarial Pass`.

---

### 2. Guardrails Architecture & Where They Are Implemented
- **Where to click in UI**: Click the **"Guardrails & Security"** tab.
- **Code Location**: `backend/app/guardrails.py`
- **Explanation for Professors**:
  > *"We implemented defense-in-depth across 4 distinct architectural boundaries:"*
  1. **Layer 1 (Input Ingestion Guardrails)**:
     - `validate_paper_text()`: Requires at least 80 characters of extractable paper text.
     - `detect_prompt_injection()`: Scans against 8 adversarial injection signatures (jailbreaks, developer mode, system prompt leaks).
     - `check_toxic_or_abusive_content()`: Filters abusive or hostile phrasing.
     - `check_academic_scope()`: Classifies whether input text contains scientific structure.
  2. **Layer 2 (Tool & MCP Boundary Guardrails)**:
     - `validate_mcp_query()`: Sanitizes query arguments against command injection and bounds results `[1, 10]`.
     - `validate_tool_return_types()`: **Data validation callback** performing strict type checking on tool outputs.
     - `safe_external_text()`: Sanitizes and truncates untrusted external API outputs to 4,000 characters.
  3. **Layer 3 (Multi-Agent & A2A Boundary Guardrails)**:
     - `sandbox_agent_context()`: Encapsulates paper text inside `<UNTRUSTED_PAPER_CONTENT>` tags so agents treat it as passive data.
     - `verify_citation_grounding()`: Prevents hallucinated citations by verifying all cited source IDs exist in the retrieved literature pool.
  4. **Layer 4 (Output Synthesis & Schema Guardrails)**:
     - `validate_score_bounds()`: Clamps Novelty (0–100), Feasibility (0–100), and Coverage (0.0–1.0).
     - `validate_verdict_consistency()`: Ensures recommendations mathematically align with calibrated scores.
     - `ReviewReport` Pydantic model: Enforces strict schema validation.

---

### 3. How Agents Interact with Tools through MCP (Clear & Interactive)
- **Where to click in UI**: Click the **"MCP Tools & Protocol"** tab.
- **Code Locations**:
  - FastMCP Server: `backend/mcp_server.py`
  - MCP Client: `backend/app/mcp_client.py`
- **What to show**:
  1. **Architecture Flow**:
     `Specialist Agent → JSON-RPC 2.0 (tools/call) → MCP Client → FastMCP Server → arXiv / OpenAlex / CrossRef → Sanitized Result → Shared State`
  2. **Executed Invocations Stream**:
     - Expand `Inspect JSON-RPC 2.0 Protocol Envelope` on any card to show the exact JSON-RPC 2.0 request and response payloads!
  3. **Live Interactive Console**:
     - Select `search_academic_literature` or `text_statistics`.
     - Click **"Execute via MCP"** right on screen during the presentation to prove the live tool interaction.

---

### 4. Context Sharing & State Updates (Requirement 1)
- **Where to click in UI**: Click the **"Context & State Updates"** tab.
- **Code Location**: `backend/app/pipeline.py`
- **What to show**:
  - The step-by-step visual sequence of how intermediate outputs from one agent feed into the next:
    - Step 1: Input Ingestion → updates `state["title"]`, `state["paper_text"]`
    - Step 2: Document Structure Extraction (FastMCP) → updates `state["stats"]`
    - Step 3: Multi-Source Literature Retrieval (FastMCP) → updates `state["sources"]`
    - Step 4: CrewAI Specialist Review → updates `state["findings"]`
    - Step 5: A2A Message Handoff → updates `state["critique"]`
    - Step 6: Synthesis Panel → updates `state["recommendation"]`

---

### 5. A2A Interoperability: CrewAI + Google ADK (Requirements 6 & 7)
- **Where to click in UI**: Click the **"A2A Inter-Agent Protocol"** tab.
- **Code Locations**:
  - A2A Boundary: `backend/app/a2a.py`
  - CrewAI Specialists: `backend/app/crew.py`
  - Google ADK Agent: `backend/app/adk_agent.py`
- **What to show**:
  - Explain the official A2A handshake at `/.well-known/agent-card.json`.
  - Show the JSON-RPC 2.0 request sent from the CrewAI specialist crew to the Google ADK Adversarial Critic agent.
  - Emphasize that **only CrewAI and Google ADK** are used in the project.

---

### 6. End-to-End Evaluation, Running Time & Cost Estimates (Requirement 10)
- **Where to click in UI**: Click the **"Evaluation & Cost"** tab.
- **Code Location**: `backend/evaluation/run_eval.py`
- **What to show**:
  - Click **"Run Benchmark Suite"** to demonstrate end-to-end evaluation over diverse scientific papers.
  - Point out the metrics:
    - **Average running time**: ~2.5 to 11.8 seconds per paper review
    - **Latency**: reported in milliseconds
    - **Token usage & Cost**: calculated in USD ($0.59 / 1M input, $0.79 / 1M output)
    - **Guardrail pass rate**: 100% verified

---

### 7. Structured Output using Pydantic (Requirement 3)
- **Where to click in UI**: In the results header:
  - Toggle **"Table View"** vs **"Card View"** to show tabular formatting.
  - Click **"Export Markdown"** to copy a structured academic Markdown report with tables.
  - Click **"Export JSON"** to copy the complete typed Pydantic ReviewReport schema.

---

## 📋 10 Requirements Summary Cheat Sheet

| Requirement | Where in Code | Where in UI |
|---|---|---|
| **1. Context Sharing & State** | `backend/app/pipeline.py:195` | Tab: "Context & State Updates" |
| **2. Two MCP Tools** | `backend/mcp_server.py`, `backend/app/mcp_client.py` | Tab: "MCP Tools & Protocol" |
| **3. Structured Output (Pydantic)** | `backend/app/schemas.py:ReviewReport` | Buttons: "Table View", "Export Markdown", "Export JSON" |
| **4. Agentic Design Patterns** | `backend/app/pipeline.py` & `backend/app/crew.py` | Tab: "Agentic Activity" |
| **5. Callbacks & Monitoring** | `backend/app/observability.py:AgentCallbackHandler` | Tab: "Critique & logs" |
| **6. A2A Protocol Communication** | `backend/app/a2a.py` | Tab: "A2A Inter-Agent Protocol" |
| **7. CrewAI + Google ADK Only** | `app/crew.py` & `app/adk_agent.py` | Badges on every turn in "Agent Debate" |
| **8. Multi-Layer Guardrails** | `backend/app/guardrails.py` | Tab: "Guardrails & Security" |
| **9. Observability & Langfuse** | `backend/app/observability.py:RunMonitor` | Pipeline event trace & Langfuse spans |
| **10. Evaluation & Cost Benchmark** | `backend/evaluation/run_eval.py` | Tab: "Evaluation & Cost" |
