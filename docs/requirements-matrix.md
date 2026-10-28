# Minimum requirements map

| Criterion | Implementation |
|---|---|
| State/context sharing | `app/pipeline.py` shared `state` dictionary carries text, sources, findings and critique |
| Two MCP tools | `mcp_server.py`: `search_arxiv` and `text_statistics` |
| Structured output | Pydantic `ReviewReport` in `app/schemas.py` |
| Agentic patterns | orchestrator + specialist reviewers + one bounded critique pass |
| Callback monitoring/logging | `app/observability.py`, event log returned in report |
| A2A interoperability | `app/a2a.py` JSON-RPC-style message boundary for ADK critic |
| CrewAI + ADK | `app/crew.py` and `app/adk_agent.py` |
| Guardrails | `app/guardrails.py`, size checks, prompt-injection warning, schema validation |
| Observability | event trace; optional Langfuse client helper |
| Evaluation | `evaluation/run_eval.py`, latency and evidence coverage |
| Cloud deployment | `Dockerfile`, `render.yaml` |

**Honest demo note:** the A2A endpoint is a minimal prototype boundary. Pair it with the matching official A2A SDK and test Agent Card/message compatibility before claiming full standards conformance. SQLite/local events are fine for a prototype; use managed PostgreSQL for durable cloud storage.
