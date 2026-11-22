import { useState } from "react";

const labels = {
  report_id: "Report ID", review_id: "Review ID", overall_recommendation: "Overall recommendation",
  sources: "Related literature", agent_conversation: "Reviewer discussion", a2a_exchanges: "Adversarial exchanges",
  mcp_invocations: "Literature tool activity", request_jsonrpc: "Tool request", response_jsonrpc: "Tool response",
};
const ordered = ["overview", "summary", "analysis", "assessment", "overall_recommendation", "novelty_score",
  "feasibility_score", "evidence_coverage", "findings", "critique_summary", "sources", "similar_papers",
  "research_gaps", "recommendations", "limitations"];
const rawFields = new Set(["request_jsonrpc", "response_jsonrpc", "request_payload", "response_payload"]);
const label = key => labels[key] || key.replaceAll("_", " ").replace(/^./, char => char.toUpperCase());

function makeReport(report) {
  const title = report.paper_title || report.title || "PaperLens report";
  const blocks = [{ kind: "title", text: title }, { kind: "text", text: report.review_id ? "Paper Review" : "Research Idea" }];
  blocks.push({ kind: "text", text: "Literature similarity estimates are based on the available titles and abstracts." });
  function render(key, value, depth) {
    if (rawFields.has(key) || value === "" || value === undefined) return;
    const name = label(key);
    if (value == null || typeof value !== "object") {
      let display = value ?? "Unavailable";
      if (typeof value === "number" && (key === "evidence_coverage" || key === "confidence")) display = `${Math.round(value * 100)}%`;
      if (key === "similarity_percentage") display = `${value}%`;
      if (key === "overall_recommendation") display = report.synthesis_recovered ? "Unavailable (specialist results recovered)" : value.replaceAll("_", " ");
      blocks.push({ kind: "text", text: `${name}: ${display}` });
      return;
    }
    if (Array.isArray(value) && !value.length) return;
    blocks.push({ kind: depth === 0 ? "heading" : "subheading", text: name });
    if (Array.isArray(value)) {
      for (const [index, entry] of value.entries()) {
        if (entry && typeof entry === "object") {
          blocks.push({ kind: "subheading", text: entry.headline || entry.title || entry.name || `${name} ${index + 1}` });
          for (const [field, item] of Object.entries(entry)) render(field, item, depth + 1);
        } else {
          blocks.push({ kind: "bullet", text: String(entry) });
        }
      }
    } else {
      for (const [field, item] of Object.entries(value)) render(field, item, depth + 1);
    }
  }
  const keys = new Set([...ordered, ...Object.keys(report)]);
  for (const key of keys) {
    if (key !== "title" && key !== "paper_title") render(key, report[key], 0);
  }
  return blocks;
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const markdownPrefixes = { title: "# ", heading: "## ", subheading: "### ", bullet: "- ", text: "" };

export default function ReportDownloads({ report, api }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const kind = report.review_id ? "paper-review" : "research-idea";
  const filename = `paperlens-${kind}-${report.review_id || report.report_id}`;

  function markdown() {
    const content = makeReport(report).map(block => markdownPrefixes[block.kind] + block.text).join("\n\n");
    download(new Blob([content + "\n"], { type: "text/markdown;charset=utf-8" }), filename + ".md");
  }

  async function pdf() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(api + "/api/reports/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blocks: makeReport(report) }),
      });
      if (!response.ok) throw new Error("Could not create the PDF. Please try again.");
      download(await response.blob(), filename + ".pdf");
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="report-downloads">
      <button type="button" className="ghost" onClick={markdown}>Download Markdown</button>
      <button type="button" className="ghost" onClick={() => download(
        new Blob([JSON.stringify(report, null, 2) + "\n"], { type: "application/json" }), filename + ".json",
      )}>Download JSON</button>
      <button type="button" className="ghost" onClick={pdf} disabled={busy}>{busy ? "Creating PDF..." : "Download PDF"}</button>
      {error && <p role="alert" className="error">{error}</p>}
    </div>
  );
}
