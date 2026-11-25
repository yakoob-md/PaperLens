# PaperLens — Multi-Agent AI Scientific Paper Review System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange.svg)](https://crewai.com)
[![Groq](https://img.shields.io/badge/Groq-Cloud%20inference-blue.svg)](https://console.groq.com/docs/models)
[![Model Context Protocol](https://img.shields.io/badge/MCP-JSON--RPC%202.0-purple.svg)](https://modelcontextprotocol.io)
[![A2A Protocol](https://img.shields.io/badge/A2A%20Protocol-Interoperable-green.svg)](https://github.com/AnanyaPandey0805/PaperLens)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)

> **PaperLens** is an Agentic AI Scientific Paper Review System that goes far beyond superficial summarization to stress-test experimental methodology, evaluate prior art differentiation across arXiv, OpenAlex, and CrossRef, enforce multi-layer guardrails, and foster multi-round debate with challenge questions.

---

## Model providers and full-text evidence

Set `GROQ_API_KEY` and `GOOGLE_API_KEY` in `backend/.env` or in your deployment's backend secrets. Groq defaults to `openai/gpt-oss-20b`. Set `GROQ_MODEL` to override it and `GROQ_AGENT_MODEL` to choose a separate agent model. `GET /api/llm-status` checks Groq credentials and model availability. Groq requests cap prompt size and completion tokens. Rate-limit retries respect Groq's retry header up to the configured wait limit.

PaperLens ranks related work by title and abstract and checks readable PDFs for the top three matches from arXiv, ACL Anthology, and OpenReview. The overlap viewer and report downloads display verified quotations and PDF page numbers. Sources that lack verified text evidence keep their abstract-only label and original similarity score. Long model inputs are capped using excerpts.

## 🌟 Key Capabilities & Architectural Highlights

1. **Paper Overview and Specialist Review**:
   - Upload a text-based PDF to see a plain-language overview and the outcomes the authors reported. Start the detailed review when ready.
   - The configured model generates analysis, scores, findings, and recommendations using the manuscript and retrieved literature.
   - Groq handles overview, abstract similarity, PDF evidence verification, idea extraction, and adversarial critique.
   - Set `GOOGLE_API_KEY` in `backend/.env` to enable Gemini for specialist review, report synthesis, idea novelty assessment, and paper chat. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`. The three CrewAI specialists each make a model call, and synthesis makes a separate call. Missing credentials or a failed cloud request trigger Groq fallback.
   - **CrewAI Specialist Agents**: Novelty & Gap Analyst, Methodology & Reproducibility Auditor, Dataset & Evidence Reviewer.
   - **Google ADK Adversarial Critic**: Independent local reviewer that challenges unsupported findings.
   - **Review Discussion**: Specialist findings and the independent critique produce targeted questions for the authors.

2. **Model Context Protocol (MCP) Integration**:
   - Built on standard FastMCP with 5 external academic tools (`text_statistics`, `search_academic_literature`, `search_arxiv`, `search_openalex`, `search_crossref`).
   - The review invokes text statistics and literature search through an SDK client session over stdio. The dashboard shows the tool name, negotiated protocol version, result, and duration.

3. **Agent-to-Agent (A2A) Protocol Boundary**:
   - The A2A server provides agent-card discovery at `/.well-known/agent-card.json` and JSON-RPC at `/a2a`.
   - Interoperable JSON-RPC 2.0 message handoff between the review orchestrator and the Google ADK critic, using findings from CrewAI specialists.

4. **Multi-Layer Defense-in-Depth Guardrails**:
   - **Layer 1 (Input)**: Rejects recognized prompt override directives before inference. Flags potentially abusive language for review.
   - **Layer 2 (Tools)**: MCP query sanitization + tool return type-checking callbacks.
   - **Layer 3 (A2A)**: Context sandboxing via `<UNTRUSTED_PAPER_CONTENT>` tags + citation grounding verification.
   - **Layer 4 (Output)**: Score bounds validation (0–100) + Pydantic schema validation.

5. **Context Sharing & State Management**:
   - Centralized `state` dictionary feeding intermediate outputs from one agent to the next across 6 distinct pipeline transitions.

6. **Task Monitoring, Callbacks & Observability**:
   - CrewAI task callbacks and ADK model callbacks capture intermediate outputs. Model and tool failures are logged.
   - `RunMonitor` with native Langfuse tracing and execution span timing.

7. **End-to-End Evaluation & Cost Estimation**:
   - Benchmark evaluation across scientific papers with latency (ms), running time (s), provider-reported token usage where available, and estimated model API charges.

8. **Research Idea Novelty Checker ("Check Research Idea")**:
   - Allows researchers to submit an idea via direct text input or document upload (**PDF** and **DOCX** supported).
   - Automatically deconstructs the research concept: problem statement, objectives, proposed methodology, key technologies, research domain, and important keywords.
   - Cross-references against authentic academic literature across **arXiv**, **OpenAlex**, and **CrossRef** with direct paper links.
   - Delivers a comprehensive **5-Section Results Dashboard**: Idea Overview, Novelty Assessment (with a 0-100 score when supported by related literature), Similar Papers, Research Gap Mapping, and Actionable Recommendations.

---

## 📋 Evaluation Requirements Mapping

| Requirement | Implementation & Code Location | Where in UI |
|---|---|---|
| **1. Context Sharing & State Updates** | `backend/app/pipeline.py` (`state` dictionary & `StateTransition`) | Tab: **Context & State Updates** |
| **2. Tool Integration using MCP** | `backend/mcp_server.py`, `backend/app/mcp_client.py` | Tab: **MCP Tools & Protocol** |
| **3. Structured Output (Pydantic)** | `backend/app/schemas.py:ReviewReport` | **Table View**, **Export Markdown**, **Export JSON** |
| **4. Agentic Design Patterns** | Orchestrator?worker review, sequential specialists, independent critic | Tab: **Agentic Activity** |
| **5. Task Monitoring & Callbacks** | `backend/app/observability.py:AgentCallbackHandler`, `backend/app/crew.py` | Tab: **Critique & logs**, Left Event Stream |
| **6. A2A Protocol Communication** | `backend/app/a2a.py` (`/.well-known/agent-card.json`, `/a2a`) | Tab: **A2A Inter-Agent Protocol** |
| **7. CrewAI and Google ADK** | `backend/app/crew.py` (CrewAI) & `backend/app/a2a.py` (ADK critic using Groq) | Agent badges on each debate turn |
| **8. Secure Agentic System (Guardrails)** | `backend/app/guardrails.py` (4 Defense Layers) | Tab: **Guardrails & Security** |
| **9. Observability & Tracing** | `backend/app/observability.py:RunMonitor` (Langfuse + Spans) | Traced in `ReviewReport.events` |
| **10. Performance, Cost & Latency Eval** | `backend/evaluation/run_eval.py` & `sample_cases.jsonl` | Tab: **Evaluation & Cost** |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- Node.js 20+
- PowerShell (Windows) or Bash (macOS/Linux)

### 1. Backend Setup & Startup
```powershell
# Navigate to backend directory
cd backend

# Activate virtual environment
# Windows:
..\..\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Copy environment template
copy .env.example .env

# Start FastAPI server
uvicorn app.main:app --port 8000 --timeout-graceful-shutdown 8
```
- API Health Check: [http://localhost:8000/health](http://localhost:8000/health)
- Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend Setup & Startup
```powershell
# Navigate to frontend directory
cd frontend

# Install packages
npm install

# Start Vite development server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📊 Benchmark Evaluation Suite

Run the automated evaluation benchmark across diverse scientific papers:
```powershell
cd backend
python -u evaluation/run_eval.py
```

The suite submits four papers to the normal review API and reports measured wall time, pass/fail reasons, model usage, and estimated provider charges. Run one case with `python -u evaluation/run_eval.py --case case-1`. The CLI exits with a failure status if any case fails its checks.

Groq GPT-OSS 20B estimates use $0.075 per million input tokens and $0.30 per million output tokens. Override them with `GROQ_INPUT_USD_PER_MILLION` and `GROQ_OUTPUT_USD_PER_MILLION`. [Groq pricing](https://console.groq.com/docs/model/openai/gpt-oss-20b) Gemini 3.5 Flash-Lite uses standard paid-tier estimates of $0.30 per million input tokens and $2.50 per million output tokens, including thinking tokens. Override the rates for your billing tier or model with `GEMINI_INPUT_USD_PER_MILLION` and `GEMINI_OUTPUT_USD_PER_MILLION`. [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing/)

## Langfuse tracing

Create API keys in your Langfuse project's settings and add these values to `backend/.env`. Use the base URL matching your project's region.

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Restart the backend and submit a sample paper. In the report, open **Critique & logs** and copy the **Langfuse trace ID**. In your Langfuse project's **Traces** view, find `paperlens_review` and match the ID. Open the trace to show stage durations, callback statuses, and model usage. The SDK batches exports, and the CLI benchmark flushes them before exit. The app's trace export sends stage names, status, timing, and model usage metadata. [Langfuse setup](https://langfuse.com/docs/observability/get-started)

## Deploy on Render

Create a Blueprint from the GitHub branch containing `render.yaml`. The Blueprint defines one free Docker web service. It builds the React app and serves it from FastAPI, so the website and API share the same HTTPS URL.

Enter `GROQ_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and the project's regional `LANGFUSE_BASE_URL` in Render's environment settings. For an existing Blueprint, add new secrets directly to the service. Once Render reports the service is live, open its URL, check `/api/llm-status`, and submit a sample review. Match the trace ID in the report to your Langfuse dashboard. [Render Blueprints](https://render.com/docs/blueprint-spec)

The free service can sleep between demos. Model API usage is billed or limited separately according to your provider plans. [Render free hosting](https://render.com/docs/free)

## A2A critic transport

The default review uses the A2A SDK to discover and call the local ADK server through an in-process ASGI HTTP transport. For a separate local service, launch it from `backend` with these PowerShell commands.

```powershell
$env:A2A_PUBLIC_URL = "http://127.0.0.1:8001"
uvicorn app.a2a:asgi_critic --host 127.0.0.1 --port 8001
```

Set `A2A_CRITIC_URL=http://127.0.0.1:8001` in the review backend's environment to use that service. Leave it empty for the single-process setup.

## 📂 Project Directory Structure

```
PaperLens/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints (/api/review, /api/chat, /a2a)
│   │   ├── pipeline.py          # Master orchestrator, state transitions, debate rounds
│   │   ├── schemas.py           # Pydantic models (ReviewReport, ChallengeQuestion, etc.)
│   │   ├── guardrails.py        # 4-layer defense-in-depth security guardrails
│   │   ├── mcp_client.py        # FastMCP JSON-RPC 2.0 client session & telemetry
│   │   ├── a2a.py               # A2A protocol endpoint and message handoff
│   │   ├── crew.py              # CrewAI specialist reviewers with task callbacks
│   │   ├── llm.py               # Gemini and Groq structured generation
│   │   ├── observability.py     # RunMonitor, Langfuse integration, callbacks
│   │   ├── chat.py              # Interactive academic copilot
│   │   └── config.py            # Environment settings and validator
│   ├── evaluation/
│   │   ├── run_eval.py          # Performance, running time, latency & cost benchmark
│   │   └── sample_cases.jsonl   # 4 diverse scientific paper test cases
│   ├── mcp_server.py            # FastMCP Server (paperlens-research-tools)
│   ├── requirements.txt         # Python dependencies
│   └── .env.example             # Environment configuration template
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React dashboard (11 distinct tabs/views)
│   │   ├── style.css            # Custom CSS for debate bubbles, cards, guardrails
│   │   └── main.jsx             # React mount
│   ├── package.json             # Frontend dependencies
│   └── vite.config.js           # Vite build configuration
├── docs/
│   ├── requirements-matrix.md   # Detailed syllabus mapping
│   └── presentation_guide.md    # Complete walkthrough guide for presentations
└── README.md                    # Project overview & documentation
```

---

## 🛡️ Security & Privacy Notice
All uploaded manuscripts and abstracts are sandboxed inside `<UNTRUSTED_PAPER_CONTENT>` tags and processed in-memory. Sensitive environment variables are kept in `.env` and are strictly excluded from version control via `.gitignore`.


### Processing steps component

The frontend uses Tailwind and Vite. Shared UI components live in `frontend/src/components/ui`, with `@/` mapped to `frontend/src` in Vite and TypeScript. This is the shadcn-style layout so components can share UI helpers and use the same import paths. Run `npm install` in `frontend` to install Motion and TypeScript, and use `npx tsc --noEmit` to check the TSX. The reasoning animation is self-contained. To add shadcn CLI-managed components later, run `npx shadcn@latest init` in `frontend` and use `@/components/ui` as the UI alias.

The progress panel displays workflow milestones from the API. Clients request the live stream with `Accept: application/x-ndjson`. The same endpoints return their standard JSON responses for other clients.





this is for the home screen 
remove the toggle 
and integarate this
and the abckground to be same throughout the enitre system on all pages 
You are given a task to integrate an existing React component in the codebase

The codebase should support:
- shadcn project structure  
- Tailwind CSS
- Typescript

If it doesn't, provide instructions on how to setup project via shadcn CLI, install Tailwind or Typescript.

Determine the default path for components and styles. 
If default path for components is not /components/ui, provide instructions on why it's important to create this folder
Copy-paste this component to /components/ui folder:
```tsx
aurora-background.tsx
"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface AuroraBackgroundProps extends React.HTMLProps<HTMLDivElement> {
  children: ReactNode;
  showRadialGradient?: boolean;
}

export const AuroraBackground = ({
  className,
  children,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) => {
  return (
    <main>
      <div
        className={cn(
          "relative flex flex-col  h-[100vh] items-center justify-center bg-zinc-50 dark:bg-zinc-900  text-slate-950 transition-bg",
          className
        )}
        {...props}
      >
        <div className="absolute inset-0 overflow-hidden">
          <div
            //   I'm sorry but this is what peak developer performance looks like // trigger warning
            className={cn(
              `
            [--white-gradient:repeating-linear-gradient(100deg,var(--white)_0%,var(--white)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--white)_16%)]
            [--dark-gradient:repeating-linear-gradient(100deg,var(--black)_0%,var(--black)_7%,var(--transparent)_10%,var(--transparent)_12%,var(--black)_16%)]
            [--aurora:repeating-linear-gradient(100deg,var(--blue-500)_10%,var(--indigo-300)_15%,var(--blue-300)_20%,var(--violet-200)_25%,var(--blue-400)_30%)]
            [background-image:var(--white-gradient),var(--aurora)]
            dark:[background-image:var(--dark-gradient),var(--aurora)]
            [background-size:300%,_200%]
            [background-position:50%_50%,50%_50%]
            filter blur-[10px] invert dark:invert-0
            after:content-[""] after:absolute after:inset-0 after:[background-image:var(--white-gradient),var(--aurora)] 
            after:dark:[background-image:var(--dark-gradient),var(--aurora)]
            after:[background-size:200%,_100%] 
            after:animate-aurora after:[background-attachment:fixed] after:mix-blend-difference
            pointer-events-none
            absolute -inset-[10px] opacity-50 will-change-transform`,

              showRadialGradient &&
                `[mask-image:radial-gradient(ellipse_at_100%_0%,black_10%,var(--transparent)_70%)]`
            )}
          ></div>
        </div>
        {children}
      </div>
    </main>
  );
};


demo.tsx
"use client";

import { motion } from "framer-motion";
import React from "react";
import { AuroraBackground } from "@/components/ui/aurora-background";

export function AuroraBackgroundDemo() {
  return (
    <AuroraBackground>
      <motion.div
        initial={{ opacity: 0.0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{
          delay: 0.3,
          duration: 0.8,
          ease: "easeInOut",
        }}
        className="relative flex flex-col gap-4 items-center justify-center px-4"
      >
        <div className="text-3xl md:text-7xl font-bold dark:text-white text-center">
          Background lights are cool you know.
        </div>
        <div className="font-extralight text-base md:text-4xl dark:text-neutral-200 py-4">
          And this, is chemical burn.
        </div>
        <button className="bg-black dark:bg-white rounded-full w-fit text-white dark:text-black px-4 py-2">
          Debug now
        </button>
      </motion.div>
    </AuroraBackground>
  );
}

```

Install NPM dependencies:
```bash
framer-motion
```

Extend existing tailwind.config.js with this code:
```js
const {
  default: flattenColorPalette,
} = require("tailwindcss/lib/util/flattenColorPalette");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    // your paths
    "./src/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      animation: {
        aurora: "aurora 60s linear infinite",
      },
      keyframes: {
        aurora: {
          from: {
            backgroundPosition: "50% 50%, 50% 50%",
          },
          to: {
            backgroundPosition: "350% 50%, 350% 50%",
          },
        },
      },
    },
  },
  plugins: [addVariablesForColors],
};

// This plugin adds each Tailwind color as a global CSS variable, e.g. var(--gray-200).
function addVariablesForColors({ addBase, theme }: any) {
  let allColors = flattenColorPalette(theme("colors"));
  let newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

```

Implementation Guidelines
 1. Analyze the component structure and identify all required dependencies
 2. Review the component's argumens and state
 3. Identify any required context providers or hooks and install them
 4. Questions to Ask
 - What data/props will be passed to this component?
 - Are there any specific state management requirements?
 - Are there any required assets (images, icons, etc.)?
 - What is the expected responsive behavior?
 - What is the best place to use this component in the app?

Steps to integrate
 0. Copy paste all the code above in the correct directories
 1. Install external dependencies
 2. Fill image assets with Unsplash stock images you know exist
 3. Use lucide-react icons for svgs or logos if component requires them
