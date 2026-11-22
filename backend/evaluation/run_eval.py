"""End-to-end performance evaluation, latency, running time, token cost, and guardrails benchmark."""
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import asyncio, json, time, sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.pipeline import run_review
from app.llm import ModelUnavailable
from app.schemas import EvaluationSummary, EvaluationCaseResult

cases_file = Path(__file__).with_name("sample_cases.jsonl")
cases = [json.loads(x) for x in cases_file.read_text(encoding="utf-8").splitlines() if x.strip()]

async def evaluate_all_cases() -> EvaluationSummary:
    out: list[EvaluationCaseResult] = []
    total_latency = 0.0

    print(f"\n================================================================================")
    print(f"PAPERLENS: END-TO-END PERFORMANCE, COST & RUNNING TIME BENCHMARK")
    print(f"Evaluating {len(cases)} diverse scientific research paper benchmarks...")
    print(f"================================================================================\n")

    for idx, c in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] Benchmarking case '{c['id']}': {c['title'][:45]}...")
        start = time.perf_counter()
        try:
            report = await run_review(c["title"], c["text"])
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            running_time_s = round(elapsed_ms / 1000, 2)
            total_latency += elapsed_ms

            # Token & cost estimation based on input size, system prompts, and multi-agent outputs
            approx_input_tokens = len(c["text"]) // 4 + 1800  # text + multi-agent system instructions
            approx_output_tokens = 950  # specialist notes + critique + summary
            passed_guardrails = sum(1 for g in report.guardrail_audits if g.status == "PASSED")

            case_res = EvaluationCaseResult(
                id=c["id"],
                title=c["title"],
                latency_ms=elapsed_ms,
                running_time_seconds=running_time_s,
                novelty_score=report.novelty_score,
                feasibility_score=report.feasibility_score,
                evidence_coverage=report.evidence_coverage,
                sources_retrieved=len(report.sources),
                findings_count=len(report.findings),
                estimated_tokens=approx_input_tokens + approx_output_tokens,
                estimated_cost_usd=0.0,
                guardrails_passed=passed_guardrails,
                mode=report.execution_mode
            )
            out.append(case_res)
            print(f"       ✓ Completed in {running_time_s}s ({elapsed_ms} ms) | Local Ollama | Guardrails: {passed_guardrails}/{len(report.guardrail_audits)} Passed")
        except ModelUnavailable:
            raise
        except Exception as e:
            print(f"       ✗ Error evaluating {c['id']}: {e}")

    summary = EvaluationSummary(
        total_cases_evaluated=len(out),
        average_latency_ms=round(total_latency / max(1, len(out)), 2),
        average_running_time_s=round(total_latency / max(1, len(out)) / 1000, 2),
        total_estimated_cost_usd=0.0,
        average_cost_per_review_usd=0.0,
        results=out
    )

    print(f"\n================================================================================")
    print(f"BENCHMARK SUMMARY RESULTS:")
    print(f"• Total Cases Evaluated   : {summary.total_cases_evaluated}")
    print(f"• Average Running Time    : {summary.average_running_time_s} seconds ({summary.average_latency_ms} ms)")
    print(f"• Total Estimated Cost    : ${summary.total_estimated_cost_usd:.6f} USD")
    print(f"• Average Cost per Review : ${summary.average_cost_per_review_usd:.6f} USD")
    print(f"================================================================================\n")

    return summary

if __name__ == "__main__":
    summary = asyncio.run(evaluate_all_cases())
    print(summary.model_dump_json(indent=2))
