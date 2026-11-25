"""Retrieve supported open PDFs and verify the model's evidence against extracted text."""

import asyncio
import io
import logging
import re
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pdfplumber
from pydantic import BaseModel, Field

from .llm import ModelUnavailable, complete
from .progress import progress
from .schemas import AbstractPassage, FullTextEvidence, FullTextPassage

logger = logging.getLogger(__name__)


def supported_pdf(*links: str | None) -> str | None:
    for link in links:
        if not link:
            continue
        try:
            url = urlsplit(link)
        except ValueError:
            continue
        if url.scheme not in {"http", "https"} or url.username or url.password:
            continue
        if url.netloc.lower() in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            match = re.fullmatch(r"/(?:abs|pdf)/((?:[0-9]{4}\.[0-9]{4,5}|[a-zA-Z.-]+/[0-9]{7})(?:v[0-9]+)?)(?:\.pdf)?", url.path)
            if match:
                return "https://arxiv.org/pdf/" + match[1]
        if url.netloc.lower() == "aclanthology.org":
            match = re.fullmatch(r"/([A-Za-z0-9][A-Za-z0-9.\-]+?)(?:\.pdf|/)?", url.path)
            if match and ("." in match[1] or "-" in match[1]):
                return f"https://aclanthology.org/{match[1]}.pdf"
        if url.netloc.lower() == "openreview.net" and url.path in {"/forum", "/pdf"}:
            ids = parse_qs(url.query).get("id", [])
            if len(ids) == 1 and re.fullmatch(r"[A-Za-z0-9_-]{4,64}", ids[0]):
                return "https://openreview.net/pdf?" + urlencode({"id": ids[0]})
        if url.netloc.lower() == "proceedings.mlr.press":
            match = re.fullmatch(r"/(v\d+)/([A-Za-z0-9-]+)(?:\.html|/\2\.pdf)", url.path)
            if match:
                volume, paper = match.groups()
                return f"https://proceedings.mlr.press/{volume}/{paper}/{paper}.pdf"
        if url.netloc.lower() == "openaccess.thecvf.com":
            match = re.fullmatch(r"/content((?:_[A-Z]+_\d{4}|/[A-Z]+\d{4}))/(?:html|papers)/([\w.-]+)\.(?:html|pdf)", url.path, re.ASCII)
            if match:
                return f"https://openaccess.thecvf.com/content{match[1]}/papers/{match[2]}.pdf"
    return None


async def fetch_pdf(url: str, max_bytes: int = 12 * 1024 * 1024, timeout: float = 42) -> bytes:
    canonical = supported_pdf(url)
    if not canonical:
        raise ValueError("This PDF host is not supported for server preview.")
    data = bytearray()
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", canonical) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError("The PDF exceeds the size limit.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("The source did not return a PDF.")
    return bytes(data)


def _passages(data: bytes, subject: str) -> list[dict]:
    terms = set(re.findall(r"[a-z][a-z-]{4,}", subject.lower())) - {
        "their", "using", "these", "study", "paper", "method", "about", "which", "between", "results",
    }
    passages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for index, page in enumerate(pdf.pages[:60], 1):
            try:
                page_text = page.extract_text(x_tolerance_ratio=0.2) or ""
            finally:
                page.close()
            if re.match(r"(?is)^\s*(?:references|bibliography)\s*(?:\n|$)", page_text):
                break
            text = " ".join(page_text.split())
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            for sentence in sentences:
                words = sentence.split()
                for start in range(0, len(words), 50):
                    chunk = " ".join(words[start:start + 50])
                    if len(chunk.split()) < 8:
                        continue
                    matched = terms & set(re.findall(r"[a-z][a-z-]{4,}", chunk.lower()))
                    passages.append((len(matched), index, chunk))
    passages.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for _, page, passage in passages:
        if sum(item["page"] == page for item in selected) == 2:
            continue
        selected.append({"passage_id": str(len(selected) + 1), "page": page, "text": passage})
        if len(selected) == 9:
            break
    return selected


def abstract_passages(abstract: str, subject: str) -> list[AbstractPassage]:
    """Locate exact source sentences that share specific terms with the submission."""
    ignored = {
        "about", "after", "across", "based", "between", "different", "method", "methods",
        "paper", "propose", "research", "results", "study", "their", "these", "using", "which",
    }
    terms = set(re.findall(r"[^\W\d_]{5,}", subject.casefold())) - ignored
    ranked = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", abstract.strip()):
        # Keep a contiguous span even if a publisher supplies an unusually long sentence.
        quote = sentence.strip()[:480].rstrip()
        shared = sorted(terms & set(re.findall(r"[^\W\d_]{5,}", quote.casefold())))
        if shared and quote not in seen:
            ranked.append(AbstractPassage(quote=quote, shared_terms=shared[:8]))
            seen.add(quote)
    return sorted(ranked, key=lambda item: -len(item.shared_terms))[:2]


class _EvidenceItem(BaseModel):
    passage_id: str
    comparison: str


class _EvidenceResult(BaseModel):
    passages: list[_EvidenceItem] = Field(max_length=3)


async def check_full_text(sources: list[dict], subject: str, *, use_model: bool = True) -> None:
    """Inspect up to three readable PDFs in similarity order."""
    shortlist = []
    for source in sources:
        url = supported_pdf(source.get("pdf_url"), source.get("url"))
        source["full_text_evidence"] = FullTextEvidence(
            status="abstract_only", pdf_url=url,
            abstract_passages=abstract_passages(source.get("abstract") or "", subject),
            note="This PDF was outside the full-text review shortlist." if url
            else "The literature index did not provide a PDF on a supported open repository.",
        )
        if url and use_model:
            shortlist.append((source, url))
        elif url:
            source["full_text_evidence"].note = "PDF comparison is unavailable in the extractive report."
    if not shortlist:
        return
    progress("Checking full-text evidence for the closest matches")

    async def retrieve(url):
        try:
            data = await fetch_pdf(url)
            return await asyncio.to_thread(_passages, data, subject)
        except Exception as exc:
            logger.info("Full-text source unavailable: %s", type(exc).__name__)
            return []

    checked = []
    start = 0
    while start < len(shortlist) and len(checked) < 3:
        batch = shortlist[start:start + (3 - len(checked))]
        start += len(batch)
        retrieved = await asyncio.gather(*(retrieve(url) for _, url in batch))
        for (source, url), passages in zip(batch, retrieved):
            if passages:
                checked.append((source, url, passages))
            else:
                source["full_text_evidence"].note = "The PDF could not be downloaded or had no readable text."
    for index, (source, url, passages) in enumerate(checked, 1):
        progress(f"Verifying PDF passages for match {index} of {len(checked)}")
        try:
            answer = await complete(
                "Select up to three candidate PDF passages that substantiate a concrete overlap or difference "
                "in research question, method, data, or findings. Return each passage_id with one sentence "
                "explaining the comparison. Use an empty list if none supports a comparison.",
                {"subject": subject[:2200], "paper_title": source["title"], "passages": passages},
                _EvidenceResult,
                provider="groq",
                activity="PDF evidence verification",
                max_output_tokens=850,
            )
        except ModelUnavailable:
            source["full_text_evidence"].note = "The model could not verify passages from this PDF."
            continue
        by_id = {item["passage_id"]: item for item in passages}
        verified = []
        for item in answer.passages:
            excerpt = by_id.pop(item.passage_id, None)
            if excerpt and item.comparison.strip():
                verified.append(FullTextPassage(page=excerpt["page"], quote=excerpt["text"], comparison=item.comparison.strip()))
        source["full_text_evidence"] = FullTextEvidence(
            status="full_text_evidence" if verified else "abstract_only",
            pdf_url=url, passages=verified,
            abstract_passages=source["full_text_evidence"].abstract_passages,
            note="Citations use PDF page numbers. The similarity score remains abstract-based."
            if verified else "The retrieved PDF excerpts did not yield a verified comparison.",
        )
