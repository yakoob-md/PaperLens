import re

from .progress import progress
from .llm import ModelUnavailable, complete, excerpt
from .observability import current_monitor
from .schemas import PaperOverview
from .guardrails import enforce_review_input


def _abstract(text: str) -> str:
    heading = re.search(r"(?im)^\s*abstract\b[\s.:–-]*", text[:5000])
    if not heading:
        return ""
    body = text[heading.end():heading.end() + 1800]
    section = re.split(
        r"\n\s*(?:\d+(?:\.\d+)*\s+\S|introduction\b|keywords\b|[∗*†‡])", body, maxsplit=1, flags=re.I,
    )[0]
    section = re.sub(r"(?<=\w)-\n(?=\w)", "-", section)
    return " ".join(section.split())


async def summarize_paper(title: str, text: str) -> PaperOverview:
    enforce_review_input(title, text)
    progress("Writing the paper overview and identifying key outcomes")
    abstract = _abstract(text)
    opening = text[:3500]
    introduction = re.search(r"(?im)^\s*(?:1\.?\s+)?introduction\s*$", opening)
    if introduction:
        opening = opening[introduction.end():]

    try:
        overview = await complete(
            "Write a reader-friendly overview in 3–5 complete sentences. Explain the problem, the authors' "
            "approach, what makes it distinctive, and why it matters. Check the opening and body excerpts "
            "for reported evaluation results. Give up to five concrete outcomes, one sentence each, naming "
            "the task or dataset and preserving reported numbers, units, and comparisons where available. "
            "Distinguish measured results from claims or proposals. Leave key_outcomes empty only when "
            "there are no identifiable results in the supplied excerpts. Supply one short literature search query.",
            {"title": title, "opening_or_abstract": abstract or opening, "body_excerpts": excerpt(text[3500:], 2000)},
            PaperOverview,
            provider="groq",
            max_output_tokens=650,
            activity="Paper overview",
        )
    except ModelUnavailable:
        note = "Model providers unavailable. Showing an excerpt from the submitted paper."
        progress(note)
        monitor = current_monitor.get()
        if monitor:
            monitor.log("paper_overview", "warning", note)
        return PaperOverview(
            about=" ".join((abstract or opening or text).split())[:1100],
            key_outcomes=[], search_query=" ".join(title.split()[:16])[:250] or " ".join(text.split()[:12])[:250],
            extractive=True,
        )
    overview.extractive = False
    if len(overview.about.split()) < 35:
        if len(abstract.split()) >= 35:
            overview.about = abstract
        else:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(opening.split()))
            preview = " ".join(sentence for sentence in sentences if 12 <= len(sentence.split()) <= 70)
            if len(preview.split()) >= 35:
                overview.about = excerpt(preview, 850)
    if not overview.key_outcomes and abstract:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", abstract)
        overview.key_outcomes = [
            sentence for sentence in sentences
            if re.search(r"\b(?:achieved?|achieves|outperformed|established?|establishes|"
                         r"(?:experiments?|results?) (?:show(?:ed)?|demonstrat(?:e|ed)))\b", sentence, re.I)
            and not re.search(r"\b(?:aim to|expect to|could|would|will)\b", sentence, re.I)
        ][:5]
    return overview
