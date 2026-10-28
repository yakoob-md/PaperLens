# PaperLens — Agentic AI Research Paper Review

A first-evaluation starter based on the supplied synopsis. It supports PDF/text input, literature lookup through MCP, specialised CrewAI review tasks, an ADK critic boundary, Pydantic output validation, guardrails, event logs, evaluation smoke tests, and a React dashboard.

## Quick start
Requires Python 3.11+ and Node 20+.

Backend:
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1 ; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Copy .env.example to .env, then:
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open the Vite URL. The API defaults to `http://localhost:8000`. The app runs in transparent demo-fallback mode without API keys. Set `GROQ_API_KEY` for CrewAI model tasks and `GOOGLE_API_KEY` for the Google ADK critic.

## Important accuracy notes
This is an MVP scaffold, not a validated scientific review product. Novelty/feasibility scores are heuristic, literature search can miss relevant work, and experimental results are not independently re-run. Verify library-version compatibility and the A2A protocol handshake before claiming standards compliance. Keep secrets in environment variables, never commit `.env`.

See `docs/requirements-matrix.md` for the professor's criteria mapped to code.
