from datetime import datetime, timezone
from time import perf_counter
from .schemas import PipelineEvent
from .config import settings

def optional_langfuse():
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

class RunMonitor:
    def __init__(self):
        self.events = []
        self.langfuse = optional_langfuse()
        self.trace = None
        if self.langfuse:
            try:
                self.trace = self.langfuse.trace(name="paperlens_review")
            except Exception:
                self.trace = None

    def log(self, stage: str, status: str, message: str, duration_ms: float | None = None):
        self.events.append(PipelineEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            status=status,
            message=str(message)[:1000],
            duration_ms=duration_ms
        ))
        if self.trace:
            try:
                self.trace.event(name=stage, metadata={"status": status, "message": message, "duration_ms": duration_ms})
            except Exception:
                pass

    def start(self, stage: str) -> float:
        self.log(stage, "started", f"{stage} started")
        return perf_counter()

    def finish(self, stage: str, start: float, message: str = "Completed") -> float:
        elapsed = round((perf_counter() - start) * 1000, 2)
        self.log(stage, "completed", message, elapsed)
        if self.trace:
            try:
                self.trace.span(name=stage, metadata={"duration_ms": elapsed, "message": message})
            except Exception:
                pass
        return elapsed
