"""Measure the normal paper review endpoint using the bundled academic cases."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

import httpx

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.observability import optional_langfuse
from app.progress import progress
from app.schemas import EvaluationCaseResult, EvaluationSummary, ReviewReport

cases_file = Path(__file__).with_name("sample_cases.jsonl")
cases = [json.loads(line) for line in cases_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def review_issues(report: ReviewReport) -> list[str]:
    issues = []
    tools = {call.tool_name for call in report.mcp_invocations if call.status == "SUCCESS"}
    if not {"text_statistics", "search_academic_literature"} <= tools:
        issues.append("MCP tools did not both complete")
    if not any(exchange.status == "SUCCESS" for exchange in report.a2a_exchanges):
        issues.append("A2A critic did not complete")
    if len({event.stage for event in report.events if event.stage.startswith("callback.") and
            event.stage.endswith(".specialist_review")}) < 3:
        issues.append("Fewer than three specialist task callbacks completed")
    if report.synthesis_recovered:
        issues.append("Synthesis fell back to specialist scores")
    if not report.model_usage:
        issues.append("No model usage recorded")
    return issues


async def evaluate_all_cases(case_id: str | None = None) -> EvaluationSummary:
    from app.main import app

    results = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://paperlens") as client:
        for case in cases:
            if case_id and case["id"] != case_id:
                continue
            started = perf_counter()
            progress(f"Evaluating {case['id']}: {case['title']}")
            print(f"Evaluating {case['id']}: {case['title']}", file=sys.stderr, flush=True)
            fields = {}
            try:
                response = await client.post("/api/review", json={"title": case["title"], "text": case["text"]})
                response.raise_for_status()
                report = ReviewReport.model_validate(response.json())
                usage = report.model_usage
                costs = [call.estimated_cost_usd for call in usage]
                fields = dict(
                    issues=review_issues(report), novelty_score=report.novelty_score,
                    feasibility_score=report.feasibility_score, evidence_coverage=report.evidence_coverage,
                    sources_retrieved=len(report.sources), findings_count=len(report.findings),
                    estimated_tokens=sum(call.input_tokens + call.output_tokens for call in usage),
                    estimated_cost_usd=sum(costs) if costs and None not in costs else None,
                    model_calls=len(usage), estimated_token_calls=sum(call.estimated for call in usage),
                    guardrails_passed=sum(audit.status == "PASSED" for audit in report.guardrail_audits),
                    mode=report.execution_mode, review_id=report.review_id,
                    langfuse_trace_id=report.langfuse_trace_id,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = exc.response.json().get("detail") if status == 503 else None
                reason = f"Review endpoint returned HTTP {status}"
                fields = {"issues": [f"{reason} — {detail}" if detail else reason], "mode": "error"}
            except Exception as exc:
                fields = {"issues": [f"Review failed: {type(exc).__name__}"], "mode": "error"}
            elapsed = perf_counter() - started
            result = EvaluationCaseResult(
                id=case["id"], title=case["title"], latency_ms=round(elapsed * 1000, 2),
                running_time_seconds=round(elapsed, 2), passed=not fields["issues"], **fields,
            )
            results.append(result)
            progress(f"{case['id']} {'passed' if result.passed else 'failed'} in {elapsed:.1f} seconds")
            print(f"{case['id']}: {'PASS' if result.passed else 'FAIL'} in {elapsed:.2f}s",
                  file=sys.stderr, flush=True)
            for issue in result.issues:
                print(f"  {issue}", file=sys.stderr, flush=True)

    count = len(results)
    costs = [result.estimated_cost_usd for result in results]
    total_cost = sum(costs) if costs and None not in costs else None
    average_ms = sum(result.latency_ms for result in results) / count if count else 0.0
    return EvaluationSummary(
        total_cases_evaluated=count, passed_cases=sum(result.passed for result in results),
        average_latency_ms=round(average_ms, 2), average_running_time_s=round(average_ms / 1000, 2),
        total_estimated_cost_usd=total_cost,
        average_cost_per_review_usd=total_cost / count if total_cost is not None else None,
        results=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case["id"] for case in cases])
    options = parser.parse_args()
    summary = asyncio.run(evaluate_all_cases(options.case))
    client = optional_langfuse()
    if client:
        client.flush()
    print(summary.model_dump_json(indent=2))
    sys.exit(summary.passed_cases != summary.total_cases_evaluated)
