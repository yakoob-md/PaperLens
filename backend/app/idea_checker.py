"""Ground idea analysis in the submitted document and live literature records."""

import asyncio
import datetime
from html import unescape
import io
import logging
import re
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
import zipfile

import httpx
import pdfplumber
from pydantic import BaseModel, Field
from typing import Literal

from .progress import progress
from .full_text import check_full_text, supported_pdf
from .config import settings
from .llm import preferred_model, complete, ModelUnavailable
from .schemas import IdeaAnalysis, IdeaNoveltyReport, NoveltyAssessment, OverlapHighlight, ResearchGap, SimilarPaper, ScoringProvenance

logger = logging.getLogger(__name__)
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_text_from_file(data: bytes, filename: str) -> str:
    """Preserve document-order paragraphs, table cells, line breaks, and text boxes."""
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n\n".join(page.extract_text(x_tolerance_ratio=0.2) or "" for page in pdf.pages).strip()
        if not text:
            raise ValueError("This PDF has no selectable text. Upload a text-based PDF or DOCX.")
        return text

    if filename.lower().endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            document = archive.getinfo("word/document.xml")
            if document.file_size > settings.max_upload_mb * 1024 * 1024:
                raise ValueError("The DOCX document body exceeds the extraction limit.")
            root = ET.fromstring(archive.read(document))
        paragraphs = []
        for paragraph in root.iter(WORD_NS + "p"):
            parts = []
            # Nested paragraphs (for example in text boxes) are emitted separately.
            stack = list(reversed(paragraph))
            while stack:
                node = stack.pop()
                if node.tag == WORD_NS + "t":
                    parts.append(node.text or "")
                elif node.tag in {WORD_NS + "tab", WORD_NS + "br", WORD_NS + "cr"}:
                    parts.append(" ")
                elif node.tag not in {WORD_NS + "p", WORD_NS + "del"}:
                    stack.extend(reversed(node))
            if line := "".join(parts).strip():
                paragraphs.append(line)
        if not paragraphs:
            raise ValueError("This DOCX has no extractable body text.")
        return "\n\n".join(paragraphs)

    raise ValueError("Upload a PDF or DOCX document.")


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


async def _llm_analysis(title: str, text: str) -> IdeaAnalysis:
    return await complete(
        "Explain the proposed idea using the submitted text. Extract its problem, objectives, methods, "
        "technologies, domain, and expected contributions. Do not invent missing details. "
        "Return up to three specific academic search phrases covering the central problem and method "
        "in important_keywords. Leave unsupported fields empty.",
        {"document_title": title, "document_text": text},
        IdeaAnalysis,
        provider="ollama",
        activity="Research idea extraction",
        max_output_tokens=650,
    )


def _reconstruct_abstract(index: dict | None) -> str:
    return " ".join(word for _, word in sorted(
        (position, word) for word, positions in (index or {}).items() for position in positions
    ))


async def _fetch_openalex(client: httpx.AsyncClient, query: str) -> list[dict]:
    response = await client.get("https://api.openalex.org/works", params={"search": query, "per-page": 12})
    response.raise_for_status()
    papers = []
    for item in response.json()["results"]:
        location = item.get("primary_location") or {}
        open_access = item.get("best_oa_location") or {}
        if item.get("title") and item.get("id"):
            papers.append(dict(
                paper_id=item["id"], title=item["title"],
                authors=[author["display_name"] for entry in item.get("authorships", [])
                         if (author := entry.get("author")) and author.get("display_name")],
                year=item.get("publication_year"), doi=item.get("doi"),
                url=location.get("landing_page_url") or item.get("doi") or item["id"],
                venue=(location.get("source") or {}).get("display_name"),
                pdf_url=(supported_pdf(open_access.get("pdf_url"), open_access.get("landing_page_url"),
                                       location.get("pdf_url"), location.get("landing_page_url"))
                         or open_access.get("pdf_url") or location.get("pdf_url")),
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
            ))
    return papers


async def _fetch_arxiv(client: httpx.AsyncClient, query: str) -> list[dict]:
    response = await client.get("https://export.arxiv.org/api/query", params={
        "search_query": " AND ".join("all:" + word for word in dict.fromkeys(re.findall(r"\w+", _normalized(query)))),
        "max_results": 10, "sortBy": "relevance",
    })
    response.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    for entry in ET.fromstring(response.text).findall("atom:entry", ns):
        paper_url = entry.findtext("atom:id", "", ns).strip().replace("http://", "https://", 1)
        title = " ".join(entry.findtext("atom:title", "", ns).split())
        if not title or not paper_url:
            continue
        pdf_url = next((link.get("href") for link in entry.findall("atom:link", ns)
                        if link.get("type") == "application/pdf"), None)
        papers.append(dict(
            paper_id=paper_url, title=title, url=paper_url, pdf_url=pdf_url,
            authors=[name.text.strip() for name in entry.findall("atom:author/atom:name", ns) if name.text],
            year=entry.findtext("atom:published", "", ns)[:4] or None,
            doi=entry.findtext("arxiv:doi", None, ns),
            abstract=" ".join(entry.findtext("atom:summary", "", ns).split()), venue="arXiv",
        ))
    return papers


async def _fetch_crossref(client: httpx.AsyncClient, query: str) -> list[dict]:
    response = await client.get("https://api.crossref.org/works", params={"query.bibliographic": query, "rows": 12})
    response.raise_for_status()
    papers = []
    for item in response.json()["message"]["items"]:
        titles = item.get("title") or []
        doi = item.get("DOI")
        if not titles or not doi:
            continue
        date = (item.get("issued", {}).get("date-parts") or [[]])[0]
        abstract = unescape(re.sub(r"<[^>]+>", " ", item.get("abstract") or "")).strip()
        abstract = re.sub(r"^abstract\s+", "", abstract, flags=re.I)
        papers.append(dict(
            paper_id=doi, title=titles[0], doi=doi, url="https://doi.org/" + doi,
            authors=[name for entry in item.get("author", [])
                     if (name := entry.get("name") or " ".join(filter(None, (entry.get("given"), entry.get("family")))))],
            year=date[0] if date else None, venue=next(iter(item.get("container-title") or []), None),
            abstract=abstract,
        ))
    return papers


class IdeaPaperAssessment(BaseModel):
    paper_id: str
    similarity_percentage: int = Field(ge=0, le=100)
    relevance_category: Literal["high", "moderate", "low"]
    overlap_summary: str
    highlights: list[OverlapHighlight] = Field(max_length=3)


class IdeaComparison(BaseModel):
    assessment: NoveltyAssessment
    papers: list[IdeaPaperAssessment] = Field(max_length=5)
    research_gaps: ResearchGap
    recommendations: list[str]


def _deduplicate(candidates: list[dict]) -> list[dict]:
    unique = []
    by_title, by_doi = {}, {}
    # Keep the abstract-rich record and reuse an open PDF from a duplicate index entry.
    for paper in sorted(candidates, key=lambda item: bool(item["abstract"]), reverse=True):
        title = " ".join(re.findall(r"\w+", _normalized(paper["title"])))
        doi = _normalized(paper.get("doi") or "").removeprefix("https://doi.org/")
        if not title:
            continue
        existing = by_title.get(title) or by_doi.get(doi)
        if existing is not None:
            if not supported_pdf(existing.get("pdf_url"), existing.get("url")):
                pdf_url = supported_pdf(paper.get("pdf_url"), paper.get("url"))
                if pdf_url:
                    existing["pdf_url"] = pdf_url
            continue
        record = dict(paper)
        unique.append(record)
        by_title[title] = record
        if doi:
            by_doi[doi] = record

    return unique


async def _search(analysis: IdeaAnalysis) -> tuple[list[dict], list[str], list[str]]:
    queries = list(dict.fromkeys(phrase.strip()[:200] for phrase in analysis.important_keywords if phrase.strip()))[:3]
    if not queries:
        return [], [], []
    sources = {"OpenAlex": _fetch_openalex, "arXiv": _fetch_arxiv, "Crossref": _fetch_crossref}
    async with httpx.AsyncClient(timeout=7.0, headers={"User-Agent": "PaperLens/1.0"}) as client:
        outcomes = await asyncio.gather(*(
            asyncio.gather(*(fetch(client, phrase) for phrase in queries), return_exceptions=True)
            for fetch in sources.values()
        ))
    succeeded, failed, candidates = [], [], []
    for name, outcome in zip(sources, outcomes):
        responses = [response for response in outcome if isinstance(response, list)]
        if responses:
            succeeded.append(name)
            for response in responses:
                candidates.extend(response)
        else:
            logger.info("Idea literature source %s unavailable: %s", name, type(outcome[0]).__name__)
            failed.append(name)
    return _deduplicate(candidates), succeeded, failed


async def check_research_idea_novelty(title: str, text: str) -> IdeaNoveltyReport:
    started = time.perf_counter()
    progress("Identifying the research question, methods, and search terms")
    analysis = await _llm_analysis(title, text)
    progress("Searching academic indexes for related work")
    candidates, succeeded, failed = await _search(analysis)
    by_id = {paper["paper_id"]: paper for paper in candidates[:15]}
    progress("Assessing novelty and comparing related work")
    context = {
        "idea": analysis.model_dump(),
        "indexes_searched": succeeded,
        "indexes_unavailable": failed,
    }
    sources = [
        {"paper_id": paper["paper_id"], "title": paper["title"][:200], "abstract": paper["abstract"][:440]}
        for paper in by_id.values()
    ]
    def validate_comparison(result):
        ids = [paper.paper_id for paper in result.papers]
        if len(ids) != len(set(ids)) or not set(ids).issubset(by_id):
            raise ValueError("Comparisons must use distinct retrieved paper IDs.")
        if not ids and result.assessment.novelty_score is not None:
            raise ValueError("Novelty requires related literature evidence.")

    result = await complete(
        "Evaluate the idea against the supplied literature records. Select up to five meaningfully related "
        "papers and use only their supplied paper_id values. Score similarity independently for each. "
        "Never claim identical work based on a title match. Only include highlights with exact quotes from "
        "the corresponding abstract, with section Abstract and page null. Include at most two highlights per paper. Assess novelty against the "
        "retrieved evidence and explain uncertainty. Return a null novelty_score if evidence is insufficient. "
        "Suggest research gaps as tentative opportunities rather than verified absences.",
        {**context, "sources": sources},
        IdeaComparison,
        provider="gemini",
        activity="Research idea novelty assessment",
        max_output_tokens=2300,
        validate=validate_comparison,
    )
    _, model, _ = preferred_model()
    progress("Verifying references and assembling the idea report")
    returned_ids = {paper.paper_id for paper in result.papers}
    if len(returned_ids) != len(result.papers) or not returned_ids.issubset(by_id):
        raise ModelUnavailable("The model did not ground its comparisons in retrieved literature. Please retry.")
    if (not candidates or not result.papers) and result.assessment.novelty_score is not None:
        raise ModelUnavailable("The model assigned a novelty score without related evidence. Please retry.")

    papers = []
    for match in sorted(result.papers, key=lambda item: item.similarity_percentage, reverse=True):
        record = by_id[match.paper_id]
        abstract = _normalized(record["abstract"])
        details = match.model_dump(exclude={"paper_id", "highlights"})
        highlights = [
            highlight.model_copy(update={"page": None, "section": "Abstract"})
            for highlight in match.highlights
            if highlight.text_snippet.strip() and _normalized(highlight.text_snippet) in abstract
        ]
        papers.append(SimilarPaper(**record, **details, highlights=highlights))

    source_records = [paper.model_dump() for paper in papers]
    await check_full_text(source_records, "\n".join([title, analysis.problem_statement,
        analysis.proposed_methodology, *analysis.research_objectives]))
    papers = [SimilarPaper.model_validate(record) for record in source_records]

    if len(text) > 10000:
        result.assessment.observations.append("The long proposal was assessed using excerpts within the local context budget.")
    if succeeded:
        result.assessment.observations.append("Indexes searched: " + ", ".join(succeeded) + ".")
    if failed:
        result.assessment.observations.append("Indexes unavailable: " + ", ".join(failed) + ".")
    return IdeaNoveltyReport(
        report_id=f"idea-{uuid.uuid4().hex[:8]}",
        title=title or text.splitlines()[0][:100],
        submitted_idea_preview=text[:600],
        analysis=analysis,
        assessment=result.assessment,
        similar_papers=papers,
        research_gaps=result.research_gaps,
        recommendations=result.recommendations,
        total_papers_searched=len(candidates),
        execution_time_seconds=round(time.perf_counter() - started, 3),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scoring_provenance=ScoringProvenance(
            methodology="LLM comparison grounded in retrieved academic records",
            similarity_engine=model,
            novelty_auditor=model,
            feasibility_auditor=model,
            audit_trail=[f"{model} compared {len(by_id)} retrieved records against the idea."],
        ),
    )
