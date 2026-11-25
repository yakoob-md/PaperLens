try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os, tempfile, pathlib
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

# Prevent CrewAI from attempting to write credentials to forbidden AppData folder on Windows:
try:
    from crewai_core import token_manager
    def _safe_storage_path() -> pathlib.Path:
        p = pathlib.Path(tempfile.gettempdir()) / "crewai_storage"
        p.mkdir(parents=True, exist_ok=True)
        return p
    token_manager.TokenManager._get_secure_storage_path = staticmethod(_safe_storage_path)
except Exception:
    pass

import re
from pydantic import ValidationError
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .llm import model_session, ModelUnavailable, GROQ_URL, groq_headers
from .overview import summarize_paper
from .progress import progress, with_progress
from .report_export import router as report_export_router
from .full_text import fetch_pdf, supported_pdf
from .schemas import (
    PaperInput, ReviewReport, ChatRequest, ChatResponse,
    McpInvokeRequest, McpInvokeResponse, EvaluationSummary,
    IdeaInput, IdeaNoveltyReport, PaperOverview
)
from .guardrails import validate_paper_text
from .pipeline import run_review
from .chat import handle_paper_chat
from .mcp_client import call_mcp_tool
from .idea_checker import check_research_idea_novelty, extract_text_from_file
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="PaperLens API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(report_export_router)

@app.middleware("http")
async def scope_model_selection(request: Request, call_next):
    with model_session():
        return await call_next(request)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "paperlens-api",
        "mode": "hybrid_llm" if settings.google_api_key.strip() else "groq",
        "active_provider": "groq",
        "active_model": settings.groq_model,
        "gemini_configured": bool(settings.google_api_key.strip()),
        "gemini_model": settings.gemini_model,
        "frameworks": ["CrewAI", "Google ADK"],
        "protocols": ["Model Context Protocol (JSON-RPC 2.0)", "A2A Protocol (JSON-RPC 2.0)"],
        "guardrail_layers": 4,
    }


@app.get("/api/llm-status")
async def get_llm_status():
    try:
        if not settings.groq_api_key.strip():
            raise ValueError("Groq key missing")
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.get(f"{GROQ_URL}/models", headers=groq_headers())
            response.raise_for_status()
            installed = [item["id"] for item in response.json().get("data", [])]
        reachable = True
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        installed = []
        reachable = False
    return {
        "active_provider": "groq",
        "active_model": settings.groq_model,
        "agent_model": settings.groq_specialist_model,
        "gemini_configured": bool(settings.google_api_key.strip()),
        "gemini_model": settings.gemini_model,
        "groq_configured": bool(settings.groq_api_key.strip()),
        "cloudflare_configured": settings.cloudflare_configured,
        "cloudflare_model": settings.cloudflare_model,
        "reachable": reachable,
        "installed_models": installed,
        "ready": reachable and {settings.groq_model, settings.groq_specialist_model}.issubset(installed),
    }


@app.post("/api/review", response_model=ReviewReport)
async def review(request: Request, payload: PaperInput):
    return await with_progress(request, lambda: run_review(
        payload.title, validate_paper_text(payload.text)
    ))

def extract_title_from_pdf_text(text: str, fallback_filename: str) -> str:
    cleaned = fallback_filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    is_arxiv_id = bool(re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", cleaned))
    is_generic = is_arxiv_id or cleaned.lower() in [
        "paper", "main", "draft", "document", "submission", "manuscript", "download", "review"
    ]
    if is_generic or len(cleaned) < 4:
        lines = [line.strip() for line in text[:1500].splitlines() if line.strip()]
        candidate_lines = []
        for line in lines[:8]:
            if re.search(r"^(arxiv:|under review|proceedings of|conference on|ieee|acm|draft|preprint)", line, re.I):
                continue
            if re.search(r"^(abstract|introduction|1\.|1\s+[A-Z])", line, re.I):
                break
            candidate_lines.append(line)
            combined = " ".join(candidate_lines)
            if len(combined.split()) >= 3 and len(combined) > 12:
                break
        if candidate_lines:
            extracted = " ".join(candidate_lines)
            if 8 <= len(extracted) <= 180:
                return extracted
    return cleaned.title()

async def read_pdf_upload(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Upload a PDF file.")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "PDF exceeds upload limit.")
    return data


async def read_review_pdf(filename: str, data: bytes) -> tuple[str, str]:
    progress("Extracting text from the paper")
    try:
        text = await run_in_threadpool(extract_text_from_file, data, filename)
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF: {exc}") from exc
    return extract_title_from_pdf_text(text, filename), validate_paper_text(text)


@app.post("/api/review/overview/pdf", response_model=PaperOverview)
async def review_pdf_overview(request: Request, file: UploadFile = File(...)):
    data = await read_pdf_upload(file)
    filename = file.filename

    async def run():
        title, text = await read_review_pdf(filename, data)
        return await summarize_paper(title, text)

    return await with_progress(request, run)


@app.post("/api/review/pdf", response_model=ReviewReport)
async def review_pdf(request: Request, file: UploadFile = File(...)):
    data = await read_pdf_upload(file)
    filename = file.filename

    async def run():
        title, text = await read_review_pdf(filename, data)
        return await run_review(title, text)

    return await with_progress(request, run)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_paper(payload: ChatRequest):
    return await handle_paper_chat(payload)

@app.post("/api/chat/pdf", response_model=ChatResponse)
async def chat_with_pdf(file: UploadFile = File(...), context: str = Form(...)):
    try:
        payload = ChatRequest.model_validate_json(context)
    except ValidationError as exc:
        raise HTTPException(422, "Invalid chat context.") from exc
    title, text = await read_review_pdf(file.filename, await read_pdf_upload(file))
    return await handle_paper_chat(payload.model_copy(update={"paper_title": title, "paper_text": text}))


@app.post("/api/mcp/invoke", response_model=McpInvokeResponse)
async def invoke_mcp_tool(payload: McpInvokeRequest):
    result, invocation = await call_mcp_tool(payload.tool_name, payload.arguments, payload.calling_agent)
    return McpInvokeResponse(invocation=invocation, result=result)

@app.get("/api/evaluation", response_model=EvaluationSummary)
async def get_evaluation_metrics(request: Request):
    """Run end-to-end performance evaluation, latency, running time, and cost estimates."""
    from evaluation.run_eval import evaluate_all_cases
    return await with_progress(request, evaluate_all_cases)

# --- Feature: Research Idea Novelty Checker Endpoints ---

@app.post("/api/idea/check", response_model=IdeaNoveltyReport)
async def check_idea(request: Request, payload: IdeaInput):
    clean_text = payload.idea_text.strip()
    if len(clean_text) < 20:
        raise HTTPException(422, "Please describe your research idea in at least 20 characters.")
    return await with_progress(request, lambda: check_research_idea_novelty(payload.title, clean_text))

@app.post("/api/idea/check/file", response_model=IdeaNoveltyReport)
async def check_idea_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file uploaded.")
    fn = file.filename.lower()
    if not (fn.endswith(".pdf") or fn.endswith(".docx")):
        raise HTTPException(415, "Unsupported file format. Please upload a .pdf or .docx document.")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the maximum upload limit ({settings.max_upload_mb} MB).")
    filename = file.filename

    async def run():
        progress("Extracting text from the proposal")
        try:
            text = await run_in_threadpool(extract_text_from_file, data, filename)
        except Exception as exc:
            raise HTTPException(422, f"Could not extract text: {exc}")

        if len(text.strip()) < 20:
            raise HTTPException(422, "Uploaded document contains insufficient extractable text (minimum 20 characters).")

        title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        return await check_research_idea_novelty(title, text)

    return await with_progress(request, run)


@app.get("/api/idea/proxy-pdf")
async def proxy_pdf(url: str):
    """Serve supported academic PDFs inline from fixed upstreams."""
    if not supported_pdf(url):
        raise HTTPException(400, "This PDF host is not supported for server preview.")
    try:
        data = await fetch_pdf(url, max_bytes=24 * 1024 * 1024, timeout=35)
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise HTTPException(502, "The PDF preview is unavailable. Open the original link instead.") from exc
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": 'inline; filename="academic-paper.pdf"'})


static_dir = os.environ.get("PAPERLENS_STATIC_DIR")
if static_dir:
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
