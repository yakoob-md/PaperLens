from typing import Literal
from pydantic import BaseModel, Field, field_validator

class PaperInput(BaseModel):
    title: str = Field(default="Untitled research paper", max_length=240)
    text: str = Field(min_length=80, max_length=80000)
    @field_validator("text")
    @classmethod
    def trim_text(cls, value):
        value = value.strip()
        if len(value) < 80: raise ValueError("Provide at least 80 characters of paper text.")
        return value

class SourcePaper(BaseModel):
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str
    abstract: str = ""
    source: str = "arXiv"

class Finding(BaseModel):
    dimension: Literal["novelty", "methodology", "dataset_evidence"]
    verdict: Literal["positive", "mixed", "concern", "unverified"]
    headline: str = Field(max_length=180)
    explanation: str = Field(max_length=3000)
    evidence_status: Literal["supported", "partially_supported", "unverified"]
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
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

class ReviewReport(BaseModel):
    review_id: str
    paper_title: str
    overall_recommendation: Literal["major_revision", "minor_revision", "promising", "insufficient_evidence"]
    summary: str
    novelty_score: int = Field(ge=0, le=100)
    feasibility_score: int = Field(ge=0, le=100)
    findings: list[Finding]
    critique_summary: str
    sources: list[SourcePaper]
    limitations: list[str]
    evidence_coverage: float = Field(ge=0, le=1)
    execution_mode: Literal["live_llm", "demo_fallback"]
    duration_ms: float
    events: list[PipelineEvent]
    agent_activities: list[AgentActivity] = Field(default_factory=list)

