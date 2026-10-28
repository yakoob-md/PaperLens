try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os
os.environ["PYTHONHTTPSVERIFY"] = "0"

try:
    import litellm
    litellm.ssl_verify = False
    litellm.telemetry = False
except Exception:
    pass

import io
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .schemas import PaperInput, ReviewReport
from .guardrails import validate_paper_text
from .pipeline import run_review
from .a2a import router as a2a_router

app = FastAPI(title="PaperLens API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(a2a_router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "paperlens-api",
        "mode": "live_llm" if settings.groq_api_key else "demo_fallback"
    }

@app.post("/api/review", response_model=ReviewReport)
async def review(payload: PaperInput):
    return await run_review(payload.title, validate_paper_text(payload.text, settings.max_text_chars))

@app.post("/api/review/pdf", response_model=ReviewReport)
async def review_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Upload a PDF file.")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "PDF exceeds upload limit.")
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise HTTPException(422, f"Could not parse PDF: {type(exc).__name__}")
    return await run_review(file.filename.rsplit(".", 1)[0], validate_paper_text(text, settings.max_text_chars))
