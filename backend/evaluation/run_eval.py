"""End-to-end performance evaluation, latency, running time, and cost estimation benchmark."""
import json, time, urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000/api/review"
cases_file = Path(__file__).with_name("sample_cases.jsonl")
cases = [json.loads(x) for x in cases_file.read_text(encoding="utf-8").splitlines() if x.strip()]

# Pricing per million tokens (Groq llama-3.3-70b / gpt-oss-120b standard rates)
INPUT_COST_PER_M = 0.59
OUTPUT_COST_PER_M = 0.79

out = []
total_latency = 0.0
total_estimated_cost = 0.0

print(f"Starting performance & cost evaluation over {len(cases)} test cases...")

for c in cases:
    payload = json.dumps({"title": c["title"], "text": c["text"]}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        total_latency += elapsed_ms
        
        # Token & cost estimates based on paper size, prompts and outputs
        approx_input_tokens = len(c["text"]) // 4 + 1500  # text + multi-agent system prompts
        approx_output_tokens = 900  # specialist notes + critique + summary
        estimated_cost_usd = round(
            (approx_input_tokens * INPUT_COST_PER_M / 1_000_000) + 
            (approx_output_tokens * OUTPUT_COST_PER_M / 1_000_000), 
            6
        )
        total_estimated_cost += estimated_cost_usd

        out.append({
            "id": c["id"],
            "title": c["title"],
            "latency_ms": elapsed_ms,
            "running_time_seconds": round(elapsed_ms / 1000, 2),
            "novelty_score": d.get("novelty_score"),
            "feasibility_score": d.get("feasibility_score"),
            "evidence_coverage": d.get("evidence_coverage"),
            "sources_retrieved": len(d.get("sources", [])),
            "findings_count": len(d.get("findings", [])),
            "estimated_tokens": approx_input_tokens + approx_output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "mode": d.get("execution_mode")
        })
    except Exception as e:
        out.append({"id": c["id"], "error": str(e)})

summary = {
    "total_cases_evaluated": len(out),
    "average_latency_ms": round(total_latency / max(1, len(out)), 2),
    "average_running_time_s": round(total_latency / max(1, len(out)) / 1000, 2),
    "total_estimated_cost_usd": round(total_estimated_cost, 6),
    "average_cost_per_review_usd": round(total_estimated_cost / max(1, len(out)), 6),
    "results": out
}

print(json.dumps(summary, indent=2))
