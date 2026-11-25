"""PaperLens critic served through the official A2A JSON-RPC transport."""

import json
from functools import lru_cache
from time import perf_counter
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill, Message, MessageSendParams,
    SendMessageRequest, SendMessageSuccessResponse, TextPart,
)
from a2a.utils import new_agent_text_message
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field

from .config import settings
from .guardrails import sandbox_agent_context
from .llm import ModelUnavailable, complete
from .observability import current_monitor
from .schemas import A2AMessageExchange, Finding, PaperOverview, SourcePaper

CARD_PATH = "/.well-known/agent-card.json"
LOCAL_URL = "http://paperlens-critic"
CRITIC_INSTRUCTION = (
    "Act as an adversarial research reviewer. Challenge unsupported findings and ask one concrete "
    "question about the strongest remaining weakness. Use at most 150 words and avoid inventing paper details."
)


class CritiqueInput(BaseModel):
    paper_title: str = Field(max_length=500)
    overview: PaperOverview
    findings: list[Finding] = Field(min_length=1, max_length=3)
    sources: list[SourcePaper] = Field(max_length=8)


class CriticModel(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        evidence = llm_request.contents[-1].parts[0].text
        answer = await complete(
            CRITIC_INSTRUCTION, sandbox_agent_context(evidence),
            provider="groq", agentic=True, activity="ADK adversarial critic", max_output_tokens=320,
        )
        monitor = current_monitor.get()
        usage = monitor.usage[-1] if monitor and monitor.usage else None
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=answer)]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.input_tokens, candidates_token_count=usage.output_tokens,
            ) if usage and usage.activity == "ADK adversarial critic" else None,
        )


def validate_critic_response(callback_context, llm_response):
    parts = llm_response.content.parts if llm_response.content else []
    text = "".join(part.text or "" for part in parts or [] if not part.thought)
    if not text.strip() or len(text) > 2500:
        raise ValueError("Critic response must contain between 1 and 2,500 characters.")
    monitor = current_monitor.get()
    if monitor:
        monitor.callbacks.on_task_intermediate(callback_context.agent_name, "critic_response", text)
    return None


class CriticExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        if len(text) > 60_000:
            raise ValueError("A2A critique input is too large.")
        evidence = CritiqueInput.model_validate_json(text)
        agent = Agent(
            name="paperlens_critic", description="Independent adversarial research reviewer",
            instruction=CRITIC_INSTRUCTION, model=CriticModel(model=settings.groq_specialist_model),
            after_model_callback=validate_critic_response,
        )
        async with InMemoryRunner(agent=agent) as runner:
            session = await runner.session_service.create_session(app_name=runner.app_name, user_id="paperlens")
            message = types.Content(role="user", parts=[types.Part(text=evidence.model_dump_json())])
            answer = ""
            async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=message):
                if event.is_final_response() and event.content:
                    answer = "".join(part.text or "" for part in event.content.parts or [] if not part.thought)
        if not answer.strip():
            raise ModelUnavailable("ADK did not return a critique.")
        await event_queue.enqueue_event(new_agent_text_message(answer.strip()))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("The critic only supports blocking message exchange.")


@lru_cache(maxsize=1)
def critic_application():
    card = AgentCard(
        name="PaperLens ADK Adversarial Critic", description="Challenges unsupported research findings.",
        protocol_version="0.3.0", url=(settings.a2a_public_url or LOCAL_URL).rstrip("/") + "/a2a",
        version="1.0.0", capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        skills=[AgentSkill(
            id="critique-findings", name="Critique findings",
            description="Stress-test the evidence supporting a paper review.", tags=["research", "peer-review"],
        )],
    )
    handler = DefaultRequestHandler(agent_executor=CriticExecutor(), task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build(rpc_url="/a2a")


async def asgi_critic(scope, receive, send):
    """ASGI entry point for optionally hosting the critic in a separate process."""
    await critic_application()(scope, receive, send)


async def execute_a2a_critique(content: str) -> tuple[str, A2AMessageExchange]:
    started = perf_counter()
    configured_url = settings.a2a_critic_url.rstrip("/")
    base_url = configured_url or LOCAL_URL
    transport = None if configured_url else httpx.ASGITransport(app=asgi_critic)
    request = SendMessageRequest(
        id=uuid4().hex,
        params=MessageSendParams(message=Message(
            role="user", message_id=uuid4().hex, parts=[TextPart(text=content)],
        )),
    )
    try:
        async with httpx.AsyncClient(transport=transport, timeout=settings.groq_timeout_seconds * 2 + 10) as http:
            card = await A2ACardResolver(http, base_url).get_agent_card()
            rpc_url = card.url if configured_url else LOCAL_URL + "/a2a"
            if httpx.URL(rpc_url).copy_with(path="/", query=None, fragment=None) != httpx.URL(base_url).copy_with(path="/", query=None, fragment=None):
                raise ValueError("The agent card must advertise an endpoint on the configured origin.")
            response = await A2AClient(http, url=rpc_url).send_message(request)
        outcome = response.root
        if not isinstance(outcome, SendMessageSuccessResponse) or not isinstance(outcome.result, Message):
            raise ModelUnavailable("The ADK critic returned an A2A error.")
        answer = "".join(p.root.text for p in outcome.result.parts if isinstance(p.root, TextPart))
        if not answer.strip():
            raise ModelUnavailable("The ADK critic returned an empty A2A message.")
    except Exception as exc:
        raise ModelUnavailable(f"A2A critic request failed ({type(exc).__name__}).") from exc

    return answer, A2AMessageExchange(
        sender_framework="CrewAI specialist reviewers", receiver_framework="Google ADK (Groq)",
        handshake_card=base_url + CARD_PATH,
        request_payload=request.model_dump(mode="json", by_alias=True, exclude_none=True),
        response_payload=response.model_dump(mode="json", by_alias=True, exclude_none=True),
        duration_ms=round((perf_counter() - started) * 1000, 2),
    )
