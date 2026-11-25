from .llm import complete
from .schemas import ChatRequest, ChatResponse
from .guardrails import enforce_no_prompt_override

async def handle_paper_chat(req: ChatRequest) -> ChatResponse:
    """Answer questions comparing the uploaded paper with a fetched research paper."""
    enforce_no_prompt_override(req.question)
    src = req.selected_source
    
    if src:
        source_context = (
            f"FETCHED RESEARCH PAPER IN FOCUS:\n"
            f"- Title: {src.title}\n"
            f"- Authors: {', '.join(src.authors) if src.authors else 'Academic Authors'} ({src.year or 'Recent'})\n"
            f"- Repository / Source: {src.source}\n"
            f"- Calculated Similarity to Uploaded Paper: {src.similarity_percentage}%\n"
            f"- Abstract: {src.abstract or 'Abstract not provided'}\n"
            f"- URL / DOI: {src.url}\n"
            f"- Relevance Notes: {src.relevance_explanation}\n"
        )
        if src.full_text_evidence and src.full_text_evidence.passages:
            source_context += "\nVerified excerpts from the source PDF:\n" + "\n".join(
                f"PDF page {passage.page}: {passage.quote}"
                for passage in src.full_text_evidence.passages
            )
    else:
        source_context = "No specific fetched paper selected; the user is asking general questions about their research."

    uploaded_context = (
        f"RESEARCHER'S UPLOADED PAPER:\n"
        f"- Title: {req.paper_title or 'User Research Paper'}\n"
        f"- Content: {req.paper_text if req.paper_text else 'Content not supplied'}\n"
    )

    system_instruction = (
        "You are PaperLens Research Copilot AI, an expert academic peer reviewer and literature research assistant. "
        "Your mission is to help the researcher understand how the fetched research paper is similar to or different from their uploaded paper, "
        "and how they can use that fetched paper for their own research. "
        "Answer questions about:\n"
        "- Methodology, model architectures, mathematical formulations, and algorithmic design\n"
        "- Data used: datasets, benchmarks, sample counts, data splits, and annotation protocols\n"
        "- Key findings, metric gains, and baseline comparisons\n"
        "- Practical utility: what the user can borrow, cite, or improve in their own research\n"
        "- Critical limitations and differences\n\n"
        "Guidelines:\n"
        "- Be academically rigorous, precise, and objective.\n"
        "- Use markdown formatting (bullet points, bold key terms, tables where appropriate).\n"
        "- Ground your answers strictly in the provided paper context.\n"
        "- Never hallucinate unmentioned citations or data."
    )

    reply = await complete(system_instruction, {
        "uploaded_paper": uploaded_context,
        "selected_source": source_context,
        "conversation": [message.model_dump() for message in req.messages[-6:]],
        "question": req.question,
    }, provider="gemini", activity="Paper chat")
    return ChatResponse(reply=reply)
