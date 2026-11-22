import json, time, uuid, datetime, re
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from .progress import progress
from .config import settings
from .llm import preferred_model, complete, model_session, ModelUnavailable
from .overview import summarize_paper
from .full_text import check_full_text
from .schemas import (
    ReviewReport, Finding, SourcePaper, AgentActivity,
    ChallengeQuestion, AgentDebateMessage, GuardrailAudit, McpToolInvocation,
    StateTransition, A2AMessageExchange, ScoringProvenance
)
from .observability import RunMonitor
from .guardrails import (
    detect_prompt_injection, audit_layer1_input, audit_layer2_tool_boundary,
    audit_layer3_a2a_context, audit_layer4_output
)
from .mcp_client import call_mcp_tool
from .crew import run_crewai_review
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("paperlens.pipeline")

class SourceAssessment(BaseModel):
    source_id: str
    similarity_percentage: int = Field(ge=0, le=100)
    relevance_explanation: str


class LiteratureAssessment(BaseModel):
    sources: list[SourceAssessment]


DIMENSIONS = ("novelty", "methodology", "dataset_evidence")
SCORE_TAGS = ("NOVELTY_SCORE", "FEASIBILITY_SCORE", "EVIDENCE_COVERAGE")
FINDING_HEADLINES = ("Novelty and prior work", "Methodology and feasibility", "Dataset and evidence")


class ReviewAnalysis(BaseModel):
    novelty_score: int = Field(ge=0, le=100)
    feasibility_score: int = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=1)
    findings: list[Finding] = Field(min_length=3, max_length=3)
    overall_recommendation: Literal["major_revision", "minor_revision", "promising", "insufficient_evidence"]
    summary: str = Field(max_length=700)
    limitations: list[str] = Field(max_length=4)

    @model_validator(mode="after")
    def check_dimensions(self):
        if {finding.dimension for finding in self.findings} != set(DIMENSIONS):
            raise ValueError("Review must include one finding per dimension")
        return self


class SpecialistFinding(BaseModel):
    verdict: Literal["positive", "mixed", "concern", "unverified"]
    headline: str = Field(max_length=180)
    explanation: str = Field(max_length=1200)
    evidence_status: Literal["supported", "partially_supported", "unverified"]
    source_ids: list[str]
    confidence: float | None = Field(ge=0, le=1)
    recommendation: str = Field(max_length=500)


class LocalReviewSynthesis(BaseModel):
    novelty_score: int = Field(ge=0, le=100)
    feasibility_score: int = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=1)
    overall_recommendation: Literal["major_revision", "minor_revision", "promising", "insufficient_evidence"]
    summary: str = Field(max_length=700)
    limitations: list[str] = Field(max_length=4)
    novelty: SpecialistFinding
    methodology: SpecialistFinding
    dataset_evidence: SpecialistFinding

    def as_review(self) -> ReviewAnalysis:
        data = self.model_dump(exclude=set(DIMENSIONS))
        return ReviewAnalysis(**data, findings=[
            Finding(dimension=dimension, **getattr(self, dimension).model_dump()) for dimension in DIMENSIONS
        ])


def recover_specialist_review(notes: list[str], source_ids: list[str], overview: str) -> ReviewAnalysis | None:
    if len(notes) != len(DIMENSIONS):
        return None

    scores = []
    findings = []
    for dimension, tag, headline, note in zip(DIMENSIONS, SCORE_TAGS, FINDING_HEADLINES, notes):
        match = re.search(rf"(?im)^\s*[-*#\s]*{tag}\**\s*:\s*\**(\d+)(?=\s|[/%%*]|$)", note)
        if not match or not 0 <= int(match[1]) <= 100:
            return None
        scores.append(int(match[1]))
        findings.append(Finding(
            dimension=dimension,
            verdict="unverified",
            headline=headline,
            explanation=note[:3000],
            evidence_status="unverified",
            source_ids=[source_id for source_id in source_ids if source_id in note],
            confidence=None,
            recommendation="",
        ))

    return ReviewAnalysis(
        novelty_score=scores[0],
        feasibility_score=scores[1],
        evidence_coverage=scores[2] / 100,
        findings=findings,
        overall_recommendation="insufficient_evidence",
        summary=overview[:700],
        limitations=["The final synthesis was unavailable. Scores and findings are from the completed specialist reviews. A combined recommendation and confidence estimates are unavailable."],
    )


def synthesis_excerpt(text: str) -> str:
    if len(text) <= 3600:
        return text
    # Sample throughout the manuscript so later methods and results survive compaction.
    anchors = (0, len(text) // 4, len(text) // 2, len(text) * 3 // 4, len(text) - 700)
    return "\n[... omitted ...]\n".join(text[index:index + 700] for index in anchors)


async def run_review(title: str, text: str) -> ReviewReport:
    with model_session():
        return await _run_review(title, text)


async def _run_review(title: str, text: str) -> ReviewReport:
    started_all = time.perf_counter()
    monitor = RunMonitor()
    overview = await summarize_paper(title, text)
    similarity_model = None
    
    # --- Req 1: Context Sharing & Shared State Dictionary ---
    state = {
        "title": title,
        "paper_text": text,
        "stats": {},
        "sources": [],
        "findings": [],
        "critique": "",
        "recommendation": ""
    }
    
    state_transitions: list[StateTransition] = []
    all_guardrail_audits: list[GuardrailAudit] = []
    all_mcp_invocations: list[McpToolInvocation] = []
    all_a2a_exchanges: list[A2AMessageExchange] = []

    now_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")

    logger.info("=" * 60)
    logger.info("PAPERLENS: MULTI-AGENT REVIEW STARTED")
    logger.info("Paper title: %s", title)
    logger.info("Input text length: %d characters", len(text))
    logger.info("=" * 60)
    
    # --------------------------------------------------------------------------
    # STEP 1: LAYER 1 GUARDRAILS & STATE INITIALIZATION (Req 1 & 8)
    # --------------------------------------------------------------------------
    l1_audits = audit_layer1_input(title, text)
    all_guardrail_audits.extend(l1_audits)
    
    has_injection, inj_msg = detect_prompt_injection(text)
    if has_injection:
        monitor.log("guardrails", "warning", f"Prompt Injection Guardrail triggered: {inj_msg}")
    monitor.log("orchestrator", "info", "Layer 1 Input Guardrails verified. Initialized shared review state.")

    state_transitions.append(StateTransition(
        step_number=1,
        stage_name="Input Ingestion & Guardrail Verification",
        source_agent="API Gateway",
        target_agent="Workflow Orchestrator",
        keys_updated=["title", "paper_text"],
        intermediate_summary=f"Validated paper input ({len(text):,} chars). Verified prompt injection & toxicity guardrails.",
        timestamp=now_ts
    ))

    # --------------------------------------------------------------------------
    # STEP 2: MCP TOOL 1 - text_statistics (Req 2 & 8)
    # --------------------------------------------------------------------------
    t_stats = monitor.start("mcp.text_statistics")
    monitor.callbacks.on_task_start("Workflow Orchestrator", "text_statistics", f"paper_text[:80]={text[:80]}")
    
    progress("Measuring the structure of the paper")
    stats_raw, inv_stats = await call_mcp_tool("text_statistics", {"paper_text": text}, "Workflow Orchestrator")
    all_mcp_invocations.append(inv_stats)
    state["stats"] = stats_raw if isinstance(stats_raw, dict) else {}
    
    monitor.finish("mcp.text_statistics", t_stats, inv_stats.response_summary)
    monitor.callbacks.on_task_finish("Workflow Orchestrator", "text_statistics", t_stats, inv_stats.response_summary)

    state_transitions.append(StateTransition(
        step_number=2,
        stage_name="Document Structure Extraction via FastMCP",
        source_agent="Workflow Orchestrator",
        target_agent="Multi-Source Literature Retriever",
        keys_updated=["stats"],
        intermediate_summary=f"Extracted {state['stats'].get('word_count', 0):,} words and {state['stats'].get('sentence_count', 0)} sentences. Fed stats into literature search.",
        timestamp=now_ts
    ))

    # --------------------------------------------------------------------------
    # STEP 3: MCP TOOL 2 - search_academic_literature (Req 2 & 8)
    # --------------------------------------------------------------------------
    t_lit = monitor.start("multi_source_literature_retrieval")
    query = overview.search_query
    monitor.callbacks.on_task_start("Multi-Source Literature Retriever", "search_academic_literature", f"query='{query}'")
    
    progress("Searching academic indexes for related papers")
    raw_sources, inv_lit = await call_mcp_tool(
        "search_academic_literature",
        {"query": query, "max_results": 7, "use_cache": False},
        "Multi-Source Literature Retriever"
    )
    all_mcp_invocations.append(inv_lit)
    
    # Layer 2 Guardrail Audit for tool execution
    tool_audits = audit_layer2_tool_boundary("search_academic_literature", query, raw_sources)
    all_guardrail_audits.extend(tool_audits)

    papers = []
    breakdown = {"arXiv": 0, "OpenAlex": 0, "CrossRef": 0}
    if isinstance(raw_sources, dict):
        papers = raw_sources.get("papers", [])
        breakdown = raw_sources.get("breakdown", breakdown)

    for i, p in enumerate(papers[:8]):
        source_name = p.get("source") or ("arXiv" if "arxiv" in str(p.get("source_id", "")).lower() else "OpenAlex" if "openalex" in str(p.get("source_id", "")).lower() else "CrossRef")
        cand_title = str(p.get("title") or "Untitled academic source")
        cand_abstract = str(p.get("abstract") or "")[:1500]
        state["sources"].append({
            "source_id": str(p.get("source_id") or f"{source_name.lower()}-{i+1}"),
            "title": cand_title,
            "authors": p.get("authors") or [],
            "year": int(p["published"]) if str(p.get("published", "")).isdigit() else None,
            "url": str(p.get("url") or "https://scholar.google.com"),
            "abstract": cand_abstract,
            "pdf_url": p.get("pdf_url"),
            "source": source_name
        })

    if state["sources"]:
        progress("Comparing the paper with retrieved literature")
        expected_ids = {source["source_id"] for source in state["sources"]}

        def validate_sources(result):
            ids = [item.source_id for item in result.sources]
            if len(ids) != len(expected_ids) or set(ids) != expected_ids:
                raise ValueError("Each supplied source ID must appear exactly once.")

        comparisons = await complete(
            "For each supplied literature record, assess its similarity to the submitted paper on a scale from 0 to 100. "
            "Use only the given titles and abstract excerpts. Give one short sentence per source and note limited evidence. "
            "Return every source ID exactly once.",
            {
                "title": title,
                "overview": overview.about,
                "outcomes": overview.key_outcomes,
                "sources": [
                    {"source_id": source["source_id"], "title": source["title"], "abstract": source["abstract"][:450]}
                    for source in state["sources"]
                ],
            },
            LiteratureAssessment,
            provider="ollama",
            activity="Abstract similarity assessment",
            max_output_tokens=750,
            validate=validate_sources,
        )
        _, similarity_model, _ = preferred_model()
        by_id = {item.source_id: item for item in comparisons.sources}
        if len(by_id) != len(comparisons.sources) or set(by_id) != {item["source_id"] for item in state["sources"]}:
            raise ModelUnavailable("The model returned an incomplete literature comparison. Please retry.")
        for source in state["sources"]:
            source.update(by_id[source["source_id"]].model_dump())

    state["sources"].sort(key=lambda x: x["similarity_percentage"], reverse=True)
    
    lit_msg = f"Retrieved {len(state['sources'])} sources across arXiv ({breakdown.get('arXiv', 0)}), OpenAlex ({breakdown.get('OpenAlex', 0)}), and CrossRef ({breakdown.get('CrossRef', 0)})."
    monitor.finish("multi_source_literature_retrieval", t_lit, lit_msg)
    monitor.callbacks.on_task_finish("Multi-Source Literature Retriever", "search_academic_literature", t_lit, lit_msg)

    state_transitions.append(StateTransition(
        step_number=3,
        stage_name="Multi-Source Literature Retrieval via FastMCP",
        source_agent="Multi-Source Literature Retriever",
        target_agent="Specialist Review",
        keys_updated=["sources"],
        intermediate_summary=f"Retrieved {len(state['sources'])} publications across arXiv, OpenAlex, and CrossRef. Transferred bibliography into shared state.",
        timestamp=now_ts
    ))

    # --------------------------------------------------------------------------
    # STEP 4: CREWAI SPECIALIST AGENTS REVIEW (Req 1, 4, 5, 7)
    # --------------------------------------------------------------------------
    t_crew = monitor.start("crewai_specialists")
    monitor.callbacks.on_task_start("CrewAI Specialists", "run_crewai_review", f"Paper: '{title}' against {len(state['sources'])} sources")
    
    progress("Reviewing novelty, methods, and evidence")
    crew = await run_crewai_review(title, text, state["sources"])
    specialist_model = (crew or {}).get("model", settings.ollama_agent_model)
    
    source_ids = [s["source_id"] for s in state["sources"]]
    progress("Synthesizing findings and scores")
    notes = [task["raw"] for task in (crew or {}).get("tasks", [])]
    synthesis_data = {
        "title": title[:240],
        "overview": overview.about[:850],
        "key_outcomes": [outcome[:220] for outcome in overview.key_outcomes[:4]],
        "paper_excerpts": synthesis_excerpt(text),
        "sources": [
            {"source_id": source["source_id"], "title": source["title"][:160], "abstract": source["abstract"][:150]}
            for source in state["sources"][:7]
        ],
        "specialist_notes": [note[:850] for note in notes[:3]],
    }
    synthesis_recovered = False
    try:
        synthesis = await complete(
            "Synthesize a concise peer review from the supplied evidence and specialist notes. "
            "Score novelty and feasibility from 0 to 100; evidence coverage is 0 to 1. "
            "Use one or two short sentences for each explanation and one for each recommendation. "
            "Confidence is 0 to 1 or null. The summary should be under 65 words. "
            "Cite only supplied IDs. State material uncertainty and mention if excerpts limit the assessment.",
            synthesis_data,
            LocalReviewSynthesis,
            provider="gemini",
            activity="Review synthesis",
            agentic=True,
            max_output_tokens=1200,
        )
        analysis = synthesis.as_review()
        _, model, _ = preferred_model()
        synthesis_note = f"{model} synthesized novelty, feasibility, evidence coverage and finding confidence."
    except ModelUnavailable:
        analysis = recover_specialist_review(notes, source_ids, overview.about)
        if analysis is None:
            raise
        synthesis_recovered = True
        synthesis_note = f"Recovered the scores and findings produced by {specialist_model} via CrewAI."
        progress("Assembling the report from completed specialist reviews")
        monitor.log("crewai_specialists", "warning", synthesis_note)

    if len(text) > 3600:
        analysis.limitations.append("The local synthesis used manuscript excerpts within its context budget.")

    findings = {finding.dimension: finding for finding in analysis.findings}
    state["findings"] = [findings[dimension] for dimension in DIMENSIONS]
    for finding in state["findings"]:
        finding.source_ids = [source_id for source_id in finding.source_ids if source_id in source_ids]

    dyn_nov = analysis.novelty_score
    dyn_feas = analysis.feasibility_score
    dyn_coverage = analysis.evidence_coverage
    mode = "live_llm"
    framework = f"CrewAI ({specialist_model})" if synthesis_recovered else model
    monitor.finish("crewai_specialists", t_crew, f"Review findings synthesized by {framework}.")
    monitor.callbacks.on_task_finish("CrewAI Specialists", "run_crewai_review", t_crew, "Model analysis complete")

    state_transitions.append(StateTransition(
        step_number=4,
        stage_name="Specialist Review",
        source_agent=framework,
        target_agent="Adversarial Critic (A2A Handoff)",
        keys_updated=["findings"],
        intermediate_summary=f"Synthesized 3 specialist findings (Novelty: {dyn_nov}/100, Feasibility: {dyn_feas}/100). Fed findings into A2A critique handoff.",
        timestamp=now_ts
    ))

    # --------------------------------------------------------------------------
    # STEP 5: LAYER 3 GUARDRAILS (A2A Context Sandboxing & Citation Verification)
    # --------------------------------------------------------------------------
    all_guardrail_audits.append(audit_layer3_a2a_context(source_ids, state["findings"]))

    # --------------------------------------------------------------------------
    # STEP 6: A2A PROTOCOL MESSAGE HANDOFF TO THE LOCAL OLLAMA CRITIC (Req 6 & 7)
    # --------------------------------------------------------------------------
    t_a2a = monitor.start("a2a_adversarial_critique")
    monitor.callbacks.on_task_start("Adversarial Critic", "a2a_adversarial_critique", f"Evaluating {len(state['findings'])} findings")
    
    from .a2a import execute_a2a_critique
    a2a_payload = json.dumps({
        "paper_title": title,
        "overview": overview.model_dump(),
        "findings": [finding.model_dump() for finding in state["findings"]],
        "sources": state["sources"][:4],
    })
    progress("Stress-testing the review findings")
    try:
        critique, a2a_record = await execute_a2a_critique(a2a_payload)
        all_a2a_exchanges.append(a2a_record)
    except ModelUnavailable:
        critique, a2a_record = "", None
        analysis.limitations.append("The independent adversarial critic was unavailable. Specialist findings and scores are available.")
        monitor.log("a2a_adversarial_critique", "warning", "The independent adversarial critic was unavailable.")


    state["critique"] = critique[:2500]
    monitor.finish("a2a_adversarial_critique", t_a2a, "Adversarial critique completed." if a2a_record else "Adversarial critique unavailable.")
    monitor.callbacks.on_task_finish("Adversarial Critic", "a2a_adversarial_critique", t_a2a, "Critique completed" if a2a_record else "Critique unavailable")

    if a2a_record:
        state_transitions.append(StateTransition(
            step_number=5,
            stage_name="A2A Inter-Agent Protocol Exchange",
            source_agent=framework,
            target_agent=a2a_record.receiver_framework,
            keys_updated=["critique"],
            intermediate_summary="Transferred structured findings across JSON-RPC 2.0 boundary. Received adversarial stress-test critique.",
            timestamp=now_ts
        ))

    # --------------------------------------------------------------------------
    # STEP 7: OVERALL RECOMMENDATION CONSENSUS (Req 1, 4)
    # --------------------------------------------------------------------------
    overall_rec = analysis.overall_recommendation
    rec_summary = analysis.summary

    state["recommendation"] = overall_rec

    state_transitions.append(StateTransition(
        step_number=6,
        stage_name="Synthesis & Pydantic Schema Validation",
        source_agent="Synthesis Panel",
        target_agent="Client Presentation Dashboard",
        keys_updated=["recommendation"],
        intermediate_summary=f"Reached consensus recommendation '{overall_rec.upper()}'. Validated output against strict Pydantic ReviewReport schema.",
        timestamp=now_ts
    ))

    # --------------------------------------------------------------------------
    # STEP 8: LAYER 4 GUARDRAILS (Output Synthesis & Schema Integrity) (Req 3, 8)
    # --------------------------------------------------------------------------
    l4_audits = audit_layer4_output(dyn_nov, dyn_feas, dyn_coverage, overall_rec)
    all_guardrail_audits.extend(l4_audits)

    # --------------------------------------------------------------------------
    # Append the actual critique to the model-generated review discussion.
    all_challenge_questions = []
    agent_conversation = [
        AgentDebateMessage(
            message_id=finding.dimension,
            round=2,
            phase="Specialist findings",
            sender_name=finding.headline,
            sender_role="Paper reviewer",
            sender_framework=framework,
            speech=finding.explanation,
            timestamp=now_ts,
        )
        for finding in state["findings"]
    ]
    for message, note in zip(agent_conversation, notes):
        challenge = re.search(r"(?im)^\s*[-*#\s]*CHALLENGE\**\s*:\s*\**(.+)$", note)
        if challenge:
            question = ChallengeQuestion(
                id=f"{message.message_id}-challenge",
                challenger_name=message.sender_name,
                challenger_role=message.sender_role,
                challenger_framework=f"CrewAI ({specialist_model})",
                target="Author",
                category=message.message_id,
                question=challenge[1].strip("* ")[:1500],
                urgency="major",
                status="open_challenge",
            )
            message.challenge_question = question
            all_challenge_questions.append(question)
    if a2a_record:
        critic_question = ChallengeQuestion(
            id="adversarial-critique",
            challenger_name="Adversarial Critic",
            challenger_role="Independent reviewer",
            challenger_framework=a2a_record.receiver_framework,
            target="Author",
            category="adversarial_critique",
            question=critique[:1500],
            urgency="major",
            status="open_challenge",
        )
        all_challenge_questions.append(critic_question)
        agent_conversation.append(AgentDebateMessage(
            message_id="adversarial-critique",
            round=3,
            phase="Adversarial critique",
            sender_name="Adversarial Critic",
            sender_role="Independent reviewer",
            sender_framework=a2a_record.receiver_framework,
            avatar_icon="shield",
            speech=critique,
            challenge_question=critic_question,
            timestamp=now_ts,
        ))
    agent_activities = [
        AgentActivity(
            agent_id=agent_id,
            name=finding.dimension.replace("_", " ").title(),
            role="Paper reviewer",
            framework=framework,
            status="completed",
            icon=icon,
            action=finding.headline,
            reasoning=finding.explanation,
            output_summary=finding.recommendation,
            metrics={"confidence": finding.confidence} if finding.confidence is not None else {},
        )
        for finding, (agent_id, icon) in zip(state["findings"], (
            ("novelty_analyst", "microscope"),
            ("methodology_reviewer", "flask"),
            ("dataset_evidence_reviewer", "gauge"),
        ))
    ]
    if a2a_record:
        agent_activities.append(AgentActivity(
            agent_id="adversarial_critic",
            name="Adversarial Critic",
            role="Independent reviewer",
            framework=a2a_record.receiver_framework,
            status="completed",
            icon="shield",
            action="Challenged findings via the A2A boundary.",
            reasoning=critique,
            output_summary=critic_question.question,
            duration_ms=a2a_record.duration_ms,
        ))

    if state["sources"]:
        subject = overview.about + "\n" + "\n".join(overview.key_outcomes) + "\n" + synthesis_excerpt(text)[:900]
        await check_full_text(state["sources"], subject)
    total_duration = round((time.perf_counter() - started_all) * 1000, 2)
    logger.info("[ORCHESTRATOR] Review completed in %.2f ms", total_duration)
    logger.info("[REPORT] Final recommendation: %s", overall_rec)
    monitor.log("report_validation", "completed", f"Review report validated successfully in {total_duration} ms.")
    
    progress("Assembling the review report")
    scoring_prov = ScoringProvenance(
        methodology="LLM review grounded in manuscript and retrieved literature",
        similarity_engine=similarity_model or "No publications to score",
        novelty_auditor=framework,
        feasibility_auditor=framework,
        audit_trail=[
            *([f"{similarity_model} scored {len(state['sources'])} retrieved publications for similarity."] if similarity_model else []),
            synthesis_note,
        ],
    )

    # --- Req 3: Pydantic Structured Output ---
    report = ReviewReport(
        review_id=str(uuid.uuid4())[:8],
        paper_title=title,
        overview=overview,
        overall_recommendation=overall_rec,
        summary=rec_summary,
        novelty_score=dyn_nov,
        feasibility_score=dyn_feas,
        findings=state["findings"],
        critique_summary=state["critique"],
        sources=[SourcePaper(**s) for s in state["sources"]],
        limitations=analysis.limitations,
        evidence_coverage=dyn_coverage,
        execution_mode=mode,
        synthesis_recovered=synthesis_recovered,
        duration_ms=total_duration,
        events=monitor.events,
        agent_activities=agent_activities,
        scoring_provenance=scoring_prov,
        # 10 Minimum Requirements Artifacts:
        challenge_questions=all_challenge_questions,
        agent_conversation=agent_conversation,
        guardrail_audits=all_guardrail_audits,
        mcp_invocations=all_mcp_invocations,
        state_transitions=state_transitions,
        a2a_exchanges=all_a2a_exchanges
    )
    return report
