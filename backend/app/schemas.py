from typing import Literal, Any
from pydantic import BaseModel, Field, field_validator

class PaperInput(BaseModel):
    title: str = Field(default="Untitled research paper", max_length=240)
    text: str = Field(min_length=80, max_length=500000)
    @field_validator("text")
    @classmethod
    def trim_text(cls, value):
        value = value.strip()
        if len(value) < 80: raise ValueError("Provide at least 80 characters of paper text.")
        return value

class PaperOverview(BaseModel):
    about: str = Field(min_length=1, max_length=2400)
    key_outcomes: list[str] = Field(max_length=8)
    search_query: str = Field(min_length=1, max_length=250)


class FullTextPassage(BaseModel):
    page: int = Field(ge=1)
    quote: str
    comparison: str


class FullTextEvidence(BaseModel):
    status: Literal["abstract_only", "full_text_evidence"]
    pdf_url: str | None = None
    passages: list[FullTextPassage] = Field(default_factory=list)
    note: str


class SourcePaper(BaseModel):
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str
    pdf_url: str | None = None
    abstract: str = ""
    source: str = "arXiv"
    similarity_percentage: int = Field(default=0, ge=0, le=100)
    relevance_explanation: str = ""
    full_text_evidence: FullTextEvidence | None = None

class Finding(BaseModel):
    dimension: Literal["novelty", "methodology", "dataset_evidence"]
    verdict: Literal["positive", "mixed", "concern", "unverified"]
    headline: str = Field(max_length=180)
    explanation: str = Field(max_length=3000)
    evidence_status: Literal["supported", "partially_supported", "unverified"]
    source_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(ge=0, le=1)
    recommendation: str = Field(max_length=1200)

class PipelineEvent(BaseModel):
    timestamp: str
    stage: str
    status: Literal["started", "completed", "warning", "error", "info"]
    message: str = Field(max_length=1000)
    duration_ms: float | None = None

class AgentActivity(BaseModel):
    agent_id: str
    name: str
    role: str
    framework: str
    status: Literal["completed", "warning", "info", "fallback"]
    icon: str = "bot"
    action: str = Field(max_length=1000)
    reasoning: str = Field(max_length=4000)
    output_summary: str = Field(max_length=1500)
    duration_ms: float | None = None
    metrics: dict[str, str | int | float] = Field(default_factory=dict)

# --- Req 1: Context Sharing & State Updates ---
class StateTransition(BaseModel):
    step_number: int
    stage_name: str
    source_agent: str
    target_agent: str
    keys_updated: list[str]
    intermediate_summary: str
    timestamp: str

# --- Req 6 & 7: A2A Inter-Agent Protocol Communication ---
class A2AMessageExchange(BaseModel):
    protocol: str = "A2A Protocol (JSON-RPC 2.0)"
    sender_framework: str = "CrewAI Specialist Reviewer"
    receiver_framework: str = "Groq Adversarial Critic"
    handshake_card: str = "/.well-known/agent-card.json"
    method: str = "message/send"
    request_payload: dict[str, Any] = Field(default_factory=dict)
    response_payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    status: str = "SUCCESS"

# --- Multi-Agent Debate & Challenge Questions ---
class ChallengeQuestion(BaseModel):
    id: str
    challenger_name: str
    challenger_role: str
    challenger_framework: str
    target: str
    category: Literal["novelty", "methodology", "dataset_evidence", "adversarial_critique"]
    question: str = Field(max_length=1500)
    urgency: Literal["critical", "major", "minor"]
    status: Literal["open_challenge", "addressed_in_rebuttal", "requires_camera_ready_ablation"]
    context_source: str = ""

class AgentDebateMessage(BaseModel):
    message_id: str
    round: int
    phase: str
    sender_name: str
    sender_role: str
    sender_framework: str
    avatar_icon: str = "bot"
    speech: str = Field(max_length=3500)
    challenge_question: ChallengeQuestion | None = None
    response_to_challenge_id: str | None = None
    timestamp: str

# --- Req 8: Guardrails Auditing ---
class GuardrailAudit(BaseModel):
    id: str
    layer: Literal["Layer 1: Input Ingestion", "Layer 2: Tool & MCP Boundary", "Layer 3: Multi-Agent & A2A Boundary", "Layer 4: Output Synthesis & Schema"]
    guardrail_name: str
    status: Literal["PASSED", "WARNING", "TRIGGERED_REMEDIATION", "BLOCKED"]
    code_location: str
    policy_description: str
    audit_findings: str

# --- Req 2: Model Context Protocol (MCP) Tool Invocations ---
class McpToolInvocation(BaseModel):
    call_id: str
    timestamp: str
    agent_name: str
    tool_name: str
    server_name: str = "paperlens-research-tools"
    protocol: str = "Model Context Protocol (JSON-RPC 2.0)"
    transport: str = "stdio / FastMCP Process"
    request_jsonrpc: dict[str, Any] = Field(default_factory=dict)
    response_summary: str
    response_jsonrpc: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    status: Literal["SUCCESS", "FALLBACK_CACHE", "FAILED"]

class ScoringProvenance(BaseModel):
    methodology: str = "Dense Semantic Embeddings + Structured Empirical Verification"
    similarity_engine: str = "Dense Vector Space Geometry (384-dim)"
    novelty_auditor: str = "Multi-Agent Claim Entailment against Academic Corpus"
    feasibility_auditor: str = "Structured Experimental Hygiene Audit (Protocol & Ablations)"
    regex_heuristics_used: bool = False
    is_pure_process_driven: bool = True
    audit_trail: list[str] = Field(default_factory=list)

class ModelUsage(BaseModel):
    provider: str
    model: str
    activity: str
    input_tokens: int
    output_tokens: int
    estimated: bool
    estimated_cost_usd: float | None


# --- Req 3: Pydantic Structured Output ---
class ReviewReport(BaseModel):
    review_id: str
    paper_title: str
    overview: PaperOverview | None = None
    overall_recommendation: Literal["major_revision", "minor_revision", "promising", "insufficient_evidence"]
    summary: str
    novelty_score: int = Field(ge=0, le=100)
    feasibility_score: int = Field(ge=0, le=100)
    findings: list[Finding]
    critique_summary: str
    sources: list[SourcePaper]
    limitations: list[str]
    evidence_coverage: float = Field(ge=0, le=1)
    synthesis_recovered: bool = False
    execution_mode: Literal["live_llm", "demo_fallback"]
    duration_ms: float
    events: list[PipelineEvent]
    model_usage: list[ModelUsage] = Field(default_factory=list)
    langfuse_trace_id: str | None = None
    agent_activities: list[AgentActivity] = Field(default_factory=list)
    scoring_provenance: ScoringProvenance = Field(default_factory=ScoringProvenance)
    
    # 10 Minimum Requirements Artifacts:
    challenge_questions: list[ChallengeQuestion] = Field(default_factory=list)
    agent_conversation: list[AgentDebateMessage] = Field(default_factory=list)
    guardrail_audits: list[GuardrailAudit] = Field(default_factory=list)
    mcp_invocations: list[McpToolInvocation] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    a2a_exchanges: list[A2AMessageExchange] = Field(default_factory=list)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    paper_title: str = Field(default="", max_length=500)
    paper_text: str = ""
    selected_source: SourcePaper | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    question: str = Field(min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    reply: str
    suggested_questions: list[str] = Field(default_factory=list)

class McpInvokeRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    calling_agent: str = "Interactive Playground Agent"

class McpInvokeResponse(BaseModel):
    invocation: McpToolInvocation
    result: Any

# --- Req 10: Evaluation & Cost Summary Schema ---
class EvaluationCaseResult(BaseModel):
    id: str
    title: str
    latency_ms: float
    running_time_seconds: float
    novelty_score: int | None = None
    feasibility_score: int | None = None
    evidence_coverage: float | None = None
    sources_retrieved: int = 0
    findings_count: int = 0
    estimated_tokens: int = 0
    estimated_cost_usd: float | None = None
    model_calls: int = 0
    estimated_token_calls: int = 0
    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    review_id: str | None = None
    langfuse_trace_id: str | None = None
    guardrails_passed: int = 0
    mode: str = "demo_fallback"

class EvaluationSummary(BaseModel):
    total_cases_evaluated: int
    passed_cases: int = 0
    average_latency_ms: float
    average_running_time_s: float
    total_estimated_cost_usd: float | None
    average_cost_per_review_usd: float | None
    results: list[EvaluationCaseResult] = Field(default_factory=list)

# --- Feature: Research Idea Novelty Checker Schemas ---
class IdeaInput(BaseModel):
    title: str = Field(default="", max_length=500)
    idea_text: str = Field(min_length=20)

class IdeaAnalysis(BaseModel):
    problem_statement: str
    research_objectives: list[str]
    proposed_methodology: str
    key_concepts_and_technologies: list[str]
    research_domain: str
    potential_contributions: list[str]
    important_keywords: list[str]

class OverlapHighlight(BaseModel):
    highlight_id: str
    text_snippet: str
    section: str = "Abstract & Core Methodology"
    page: int | None = 1
    category: str = "Core Methodology"
    matched_idea_aspect: str
    explanation: str
    differentiation_strategy: str
    severity: Literal["high", "medium", "low"] = "high"

class SimilarPaper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: str | int | None = None
    abstract: str
    venue: str | None = None
    doi: str | None = None
    url: str
    pdf_url: str | None = None
    similarity_percentage: int = Field(ge=0, le=100)
    relevance_category: Literal["high", "moderate", "low"]
    overlap_summary: str
    highlights: list[OverlapHighlight] = Field(default_factory=list)
    full_text_preview: str | None = None
    full_text_evidence: FullTextEvidence | None = None

class ResearchGap(BaseModel):
    covered_areas: list[str]
    underexplored_areas: list[str]
    suggested_directions: list[str]
    differentiation_opportunities: list[str]

class NoveltyAssessment(BaseModel):
    category: str = "Related literature found"
    novelty_score: int | None = Field(default=None, ge=0, le=100)
    explanation: str
    observations: list[str]
    explored_aspects: list[str]
    similar_aspects: list[str]
    distinctive_opportunities: list[str]
    disclaimer: str

class IdeaNoveltyReport(BaseModel):
    report_id: str
    title: str
    submitted_idea_preview: str
    analysis: IdeaAnalysis
    assessment: NoveltyAssessment
    similar_papers: list[SimilarPaper]
    research_gaps: ResearchGap
    recommendations: list[str]
    total_papers_searched: int
    execution_time_seconds: float
    timestamp: str
    scoring_provenance: ScoringProvenance = Field(default_factory=ScoringProvenance)
