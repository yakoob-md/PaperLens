"""Observability, Tracing, Task Monitoring, and Callbacks.

Provides full provenance, execution flow monitoring, and callback hooks for
CrewAI, Ollama, and MCP tools, with optional Langfuse tracing integration.
"""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from .schemas import PipelineEvent
from .config import settings

def optional_langfuse():
    """Initialize Langfuse client if API keys are configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )
    except Exception:
        return None

class AgentCallbackHandler:
    """Callback monitoring mechanism tracking intermediate outputs, execution flows, and errors."""
    
    def __init__(self, monitor: "RunMonitor"):
        self.monitor = monitor
        self.callback_logs: list[dict[str, Any]] = []

    def on_task_start(self, agent_name: str, task_name: str, inputs_summary: str = ""):
        self.monitor.log(
            stage=f"agent.{agent_name}.{task_name}",
            status="started",
            message=f"Agent '{agent_name}' initiated task '{task_name}'. Inputs: {inputs_summary[:120]}"
        )
        return perf_counter()

    def on_task_intermediate(self, agent_name: str, task_name: str, intermediate_preview: str):
        record = {
            "agent": agent_name,
            "task": task_name,
            "intermediate_preview": intermediate_preview[:200],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.callback_logs.append(record)
        self.monitor.log(
            stage=f"callback.{agent_name}",
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
        self.langfuse = optional_langfuse()
        self.trace = None
        self.callbacks = AgentCallbackHandler(self)
        
        if self.langfuse:
            try:
                self.trace = self.langfuse.trace(
                    name="paperlens_review",
                    metadata={"frameworks": "CrewAI+ADK", "protocol": "A2A+MCP"}
                )
            except Exception:
                self.trace = None

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
                self.trace.event(
                    name=stage,
                    metadata={"status": status, "message": message, "duration_ms": duration_ms}
                )
            except Exception:
                pass

    def start(self, stage: str) -> float:
        self.log(stage, "started", f"{stage} stage initiated")
        return perf_counter()

    def finish(self, stage: str, start: float, message: str = "Completed") -> float:
        elapsed = round((perf_counter() - start) * 1000, 2)
        self.log(stage, "completed", message, elapsed)
        if self.trace:
            try:
                self.trace.span(
                    name=stage,
                    metadata={"duration_ms": elapsed, "message": message}
                )
            except Exception:
                pass
        return elapsed
