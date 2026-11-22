"""Stream public workflow milestones and the validated result to opted-in clients."""

import asyncio
import json
import logging
from contextvars import ContextVar

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

MEDIA_TYPE = "application/x-ndjson"
logger = logging.getLogger(__name__)
_events: ContextVar[tuple[asyncio.Queue, asyncio.AbstractEventLoop] | None] = ContextVar("progress_events", default=None)


def progress(label: str) -> None:
    stream = _events.get()
    if stream is not None:
        queue, loop = stream
        event = {"type": "step", "label": label}
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            queue.put_nowait(event)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(queue.put_nowait, event)


async def with_progress(request: Request, operation):
    from .llm import ModelUnavailable
    if MEDIA_TYPE not in request.headers.get("accept", ""):
        return await operation()

    async def events():
        queue = asyncio.Queue()

        async def run():
            token = _events.set((queue, asyncio.get_running_loop()))
            try:
                result = await operation()
                queue.put_nowait({"type": "result", "data": result.model_dump(mode="json")})
            except HTTPException as exc:
                queue.put_nowait({"type": "error", "detail": exc.detail})
            except ModelUnavailable as exc:
                queue.put_nowait({"type": "error", "detail": str(exc)})
            except Exception:
                logger.exception("Streaming analysis failed")
                queue.put_nowait({"type": "error", "detail": "Analysis could not be completed. Please try again."})
            finally:
                _events.reset(token)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield "\n"
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(events(), media_type=MEDIA_TYPE, headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
