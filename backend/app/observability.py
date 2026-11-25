"""Observability, Tracing, Task Monitoring, and Callbacks.

Provides full provenance, execution flow monitoring, and callback hooks for
CrewAI, Groq, and MCP tools, with optional Langfuse tracing integration.
"""

from datetime import datetime, timezone
from time import perf_counter
from contextvars import ContextVar
from functools import lru_cache
import logging
from .schemas import ModelUsage, PipelineEvent
from .config import settings

logger = logging.getLogger(__name__)
current_monitor: ContextVar["RunMonitor | None"] = ContextVar("review_monitor", default=None)

@lru_cache(maxsize=1)
def optional_langfuse():
    """Initialize Langfuse client if API keys are configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url or settings.langfuse_host,
        )
    except Exception:
        logger.exception("Could not initialize Langfuse")
        return None

class AgentCallbackHandler:
    """Callback monitoring mechanism tracking intermediate outputs, execution flows, and errors."""
    
    def __init__(self, monitor: "RunMonitor"):
        self.monitor = monitor

    def on_task_start(self, agent_name: str, task_name: str, inputs_summary: str = ""):
        self.monitor.log(
            stage=f"agent.{agent_name}.{task_name}",
            status="started",
            message=f"Agent '{agent_name}' initiated task '{task_name}'. Inputs: {inputs_summary[:120]}"
        )
        return perf_counter()

    def on_task_intermediate(self, agent_name: str, task_name: str, intermediate_preview: str):
        self.monitor.log(
            stage=f"callback.{agent_name}.{task_name}",
            status="info",
            message=f"Intermediate execution flow captured: {intermediate_preview[:150]}"
        )

    def on_task_finish(self, agent_name: str, task_name: str, start_time: float, output_summary: str):
        elapsed = round((perf_counter() - start_time) * 1000, 2)
        self.monitor.log(
            stage=f"agent.{agent_name}.{task_name}",
            status="completed",
            message=f"Agent '{agent_name}' finished task. Output: {output_summary[:180]}",
            duration_ms=elapsed
        )
        return elapsed

    def on_task_error(self, agent_name: str, task_name: str, start_time: float, error: Exception):
        elapsed = round((perf_counter() - start_time) * 1000, 2)
        self.monitor.log(
            stage=f"agent.{agent_name}.{task_name}",
            status="error",
            message=f"Task error encountered: {type(error).__name__}: {str(error)[:150]}",
            duration_ms=elapsed
        )
        return elapsed

class RunMonitor:
    """Central observability tracer capturing events, timing, and Langfuse audit traces."""
    
    def __init__(self):
        self.events: list[PipelineEvent] = []
        self.usage: list[ModelUsage] = []
        self.langfuse = optional_langfuse()
        self.trace = None
        self.trace_id = None
        self.spans = {}
        self.callbacks = AgentCallbackHandler(self)
        
        if self.langfuse:
            try:
                self.trace = self.langfuse.start_observation(
                    name="paperlens_review",
                    as_type="span",
                )
            except Exception:
                logger.exception("Could not start Langfuse review trace")
            else:
                self.trace_id = self.trace.trace_id

    def log(self, stage: str, status: str, message: str, duration_ms: float | None = None):
        event = PipelineEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            status=status,
            message=str(message)[:1000],
            duration_ms=duration_ms
        )
        self.events.append(event)
        
        if self.trace:
            try:
                self.trace.start_observation(
                    name=stage,
                    as_type="event",
                    metadata={"status": status, "duration_ms": duration_ms},
                    level="ERROR" if status == "error" else "DEFAULT",
                )
            except Exception:
                logger.exception("Could not record Langfuse event")

    def start(self, stage: str) -> float:
        self.log(stage, "started", f"{stage} stage initiated")
        started = perf_counter()
        if self.trace:
            try:
                self.spans[stage] = self.trace.start_observation(name=stage, as_type="span")
            except Exception:
                logger.exception("Could not start Langfuse stage")
        return started

    def finish(self, stage: str, start: float, message: str = "Completed", status: str = "completed") -> float:
        elapsed = round((perf_counter() - start) * 1000, 2)
        self.log(stage, status, message, elapsed)
        span = self.spans.pop(stage, None)
        if span:
            try:
                span.update(metadata={"status": status, "duration_ms": elapsed}).end()
            except Exception:
                logger.exception("Could not finish Langfuse stage")
        return elapsed

    def close(self, status: str, review_id: str | None = None):
        if self.trace:
            try:
                for span in self.spans.values():
                    span.update(level="ERROR", status_message="Stage interrupted").end()
                self.trace.update(
                    level="DEFAULT" if status == "completed" else "ERROR",
                    metadata={"status": status, "review_id": review_id},
                ).end()
            except Exception:
                logger.exception("Could not close Langfuse review trace")

    def record_usage(self, provider, model, activity, input_tokens, output_tokens, estimated=False):
        cost = 0.0
        if provider == "gemini":
            rates = (settings.gemini_input_usd_per_million, settings.gemini_output_usd_per_million)
            if rates == (None, None) and model == "gemini-3.5-flash-lite":
                rates = (0.30, 2.50)
            cost = None if None in rates else (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        elif provider == "groq":
            rates = (settings.groq_input_usd_per_million, settings.groq_output_usd_per_million)
            if rates == (None, None) and model == "openai/gpt-oss-20b":
                rates = (0.075, 0.30)
            cost = None if None in rates else (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        self.usage.append(ModelUsage(
            provider=provider, model=model, activity=activity, input_tokens=input_tokens,
            output_tokens=output_tokens, estimated=estimated, estimated_cost_usd=cost,
        ))
        if self.trace:
            try:
                self.trace.start_observation(
                    name=activity, as_type="generation", model=model,
                    usage_details={"input": input_tokens, "output": output_tokens},
                    cost_details={"total": cost} if cost is not None else None,
                    metadata={"provider": provider, "token_counts_estimated": estimated},
                ).end()
            except Exception:
                logger.exception("Could not record Langfuse model usage")
