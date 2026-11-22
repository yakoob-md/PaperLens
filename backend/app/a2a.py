"""A2A Inter-Agent Communication Protocol (JSON-RPC 2.0) Boundary.

Enables interoperable agent-to-agent communication between CrewAI specialist agents
and the Ollama Adversarial Critic agent.
"""

import json, time
from fastapi import APIRouter
from pydantic import BaseModel, Field
from .llm import preferred_model, complete, ModelUnavailable
from .schemas import A2AMessageExchange

router = APIRouter()

class A2AMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int = 1
    method: str = "message/send"
    params: dict = Field(default_factory=dict)

@router.get("/.well-known/agent-card.json")
def agent_card():
    """Publish standard A2A Agent Card describing capabilities and skills."""
    return {
        "name": "PaperLens Ollama Adversarial Critic",
        "description": "Independent adversarial critic agent built with Ollama to challenge research findings",
        "url": "/a2a",
        "version": "0.2.0",
        "framework": "Ollama",
        "capabilities": {"streaming": False, "adversarial_critique": True},
        "skills": [
            {
                "id": "critique-findings",
                "name": "Critique Review Findings",
                "description": "Stress-test empirical claims, challenge novelty overlap, and interrogate experimental hygiene."
            }
        ]
    }

async def execute_a2a_critique(content: str) -> tuple[str, A2AMessageExchange]:
    """Execute adversarial critique over the A2A protocol boundary to the Ollama agent."""
    t0 = time.perf_counter()
    try:
        evidence = json.loads(content)
    except ValueError:
        evidence = content
    answer = await complete(
        "Act as an adversarial research reviewer. Challenge unsupported conclusions in the supplied "
        "findings. Ask one concrete question about the strongest remaining weakness. "
        "Use at most 150 words and avoid inventing methodological flaws or paper details.",
        evidence,
        provider="ollama",
        agentic=True,
        activity="Adversarial Critic",
        max_output_tokens=320,
    )
    _, model, _ = preferred_model()
    framework = f"Ollama ({model})"

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    # Construct A2A Message Exchange Audit Record
    parsed_input = {}
    try:
        parsed_input = json.loads(content) if isinstance(content, str) else {"raw": str(content)}
    except Exception:
        parsed_input = {"text_summary": content[:200]}

    a2a_record = A2AMessageExchange(
        protocol="A2A Protocol (JSON-RPC 2.0)",
        sender_framework="CrewAI Specialist Reviewer Crew",
        receiver_framework=framework,
        handshake_card="/.well-known/agent-card.json",
        method="message/send",
        request_payload={
            "jsonrpc": "2.0",
            "id": "a2a-handoff-1",
            "method": "message/send",
            "params": {
                "recipient": "paperlens_adversarial_critic",
                "message": parsed_input
            }
        },
        response_payload={
            "jsonrpc": "2.0",
            "id": "a2a-handoff-1",
            "result": {
                "critique": answer[:2500]
            }
        },
        duration_ms=duration_ms,
        status="SUCCESS"
    )

    return answer[:2500], a2a_record

@router.post("/a2a")
async def a2a_message(msg: A2AMessage):
    content = str(msg.params.get("message", msg.params.get("text", "")))[:10000]
    try:
        answer, _ = await execute_a2a_critique(content)
    except ModelUnavailable as exc:
        return {"jsonrpc": "2.0", "id": msg.id, "error": {"code": -32000, "message": str(exc)}}
    return {
        "jsonrpc": "2.0",
        "id": msg.id,
        "result": {"message": {"role": "agent", "parts": [{"type": "text", "text": answer}]}}
    }
