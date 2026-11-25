import ReportDownloads from "./ReportDownloads.jsx";
import { TubelightNavbar, workflowRoutes } from "@/components/ui/tubelight-navbar";
import AnalysisProgress from "./AnalysisProgress.jsx";
import { requestAnalysis } from "./requestAnalysis.js";
import React, { useEffect, useState } from "react";
import Dropzone, { pdfFiles, proposalFiles } from "./Dropzone.jsx";
import { ResearchTree, PaperViewer, EvidenceBadge } from "./ResearchResults.jsx";
import {
  Microscope, FileText, Sparkles, BookOpen, ShieldCheck,
  Gauge, Search, Activity, ArrowUpRight,
  LoaderCircle, X, FlaskConical, Workflow, Bot,
  MessageSquare, Send, Terminal, Code2, Play, RefreshCw,
  UserCheck, CheckCircle, ShieldAlert, Cpu, Database, Table,
  Lightbulb, FileSearch
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "";

const sample = `Abstract. We propose a lightweight transformer-based approach for scientific document classification. We evaluate on a public dataset of 2,400 abstracts and compare against TF-IDF and a pretrained transformer. The proposed model achieves 4.2% higher macro-F1 than TF-IDF, but only a 0.6% gain over the pretrained transformer. We use a single train-test split and do not report confidence intervals or multiple random seeds. The dataset contains computer science papers from 2019 to 2023. We do not include an ablation study to isolate the contribution of domain-specific embeddings. Methods. The dataset is split into 80% training and 20% testing. We use Adam with learning rate 0.0001 for 10 epochs. Limitations. Our study lacks external validation and ablation analysis.`;

function workflowFromLocation() {
  const path = window.location.pathname.replace(/\/$/, "");
  return workflowRoutes.find(route => route.href === path)?.id ?? "review";
}

function PaperOverview({ overview }) {
  return (
    <section className="paper-overview" aria-label="Paper overview">
      <h2>What this paper is about</h2>
      <p>{overview.about}</p>
      <h3>Key outcomes</h3>
      {overview.key_outcomes.length
        ? <ul>{overview.key_outcomes.map((outcome, index) => <li key={index}>{outcome}</li>)}</ul>
        : <p>The overview did not identify any reported outcomes in the extracted text.</p>}
    </section>
  );
}

function Badge({ children, tone = "" }) {
  return <span className={"badge " + tone}>{children}</span>;
}

export default function App() {
  // Global Mode: "review" (Paper Review) | "idea" (Check Research Idea)
  const [appMode, setAppMode] = useState(workflowFromLocation);

  // Research Idea Novelty Checker state
  const [ideaText, setIdeaText] = useState("");
  const [ideaFile, setIdeaFile] = useState(null);
  const [ideaReport, setIdeaReport] = useState(null);
  const [ideaBusy, setIdeaBusy] = useState(false);
  const [ideaSteps, setIdeaSteps] = useState([]);
  const [ideaError, setIdeaError] = useState("");

  // Interactive Paper PDF & Overlap Inspector Modal state
  const [selectedPaperForViewer, setSelectedPaperForViewer] = useState(null);

  const [file, setFile] = useState(null);
  const [report, setReport] = useState(null);
  const [overview, setOverview] = useState(null);
  const [overviewBusy, setOverviewBusy] = useState(false);
  const [overviewSteps, setOverviewSteps] = useState([]);
  const [reviewSteps, setReviewSteps] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("Findings");
  const [viewTableMode, setViewTableMode] = useState(false);

  // Filter state for Agent Debate
  const [debateFilter, setDebateFilter] = useState("all");

  // Interactive MCP Playground state
  const [playgroundTool, setPlaygroundTool] = useState("search_academic_literature");
  const [playgroundQuery, setPlaygroundQuery] = useState("transformer scientific document classification");
  const [playgroundBusy, setPlaygroundBusy] = useState(false);
  const [playgroundResult, setPlaygroundResult] = useState(null);

  // Evaluation state
  const [evalSummary, setEvalSummary] = useState(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalStep, setEvalStep] = useState("");

  // AI Chatbot state
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [selectedSourceForChat, setSelectedSourceForChat] = useState(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([
    "How should I address the agents' challenge questions?",
    "Where are the 4 layers of Guardrails implemented?",
    "How do the agents interact with tools through MCP?",
    "Explain the A2A interoperability between CrewAI and ADK"
  ]);

  async function submitIdea() {
    if (ideaBusy || (!ideaFile && ideaText.trim().length < 20)) return;
    setIdeaBusy(true); setIdeaError(""); setIdeaReport(null);
    setIdeaSteps([ideaFile ? "Uploading the proposal" : "Submitting the research idea"]);
    const onStep = step => setIdeaSteps(steps => [...steps, step]);
    try {
      if (ideaFile) {
        const form = new FormData();
        form.append("file", ideaFile);
        setIdeaReport(await requestAnalysis(API + "/api/idea/check/file", {
          method: "POST", body: form,
        }, onStep));
      } else {
        setIdeaReport(await requestAnalysis(API + "/api/idea/check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ idea_text: ideaText }),
        }, onStep));
      }
    } catch (e) {
      setIdeaError(e.message);
    } finally {
      setIdeaBusy(false);
    }
  }

  function resetIdea() {
    setIdeaReport(null);
    setIdeaSteps([]);
    setIdeaText("");
    setIdeaFile(null);
    setIdeaError("");
  }

  async function uploadPaper(selected) {
    setFile(selected);
    setReport(null);
    setOverview(null);
    setError("");
    setOverviewBusy(true);
    setOverviewSteps(["Uploading the paper"]);
    setReviewSteps([]);
    try {
      const form = new FormData();
      form.append("file", selected);
      setOverview(await requestAnalysis(API + "/api/review/overview/pdf", {
        method: "POST", body: form,
      }, step => setOverviewSteps(steps => [...steps, step])));
    } catch (e) {
      setError(e.message);
    } finally {
      setOverviewBusy(false);
    }
  }

  async function submit() {
    setBusy(true); setError(""); setReport(null);
    setReviewSteps(["Submitting the paper for review"]);
    const onStep = step => setReviewSteps(steps => [...steps, step]);
    try {
      const form = new FormData();
      form.append("file", file);
      const d = await requestAnalysis(API + "/api/review/pdf", { method: "POST", body: form }, onStep);
      setReport(d);
      if (d.sources?.length) {
        setSelectedSourceForChat(d.sources[0]);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setReport(null); setFile(null); setError(""); setOverview(null);
    setOverviewSteps([]); setReviewSteps([]);
    setChatMessages([]); setChatInput(""); setSelectedSourceForChat(null);
    setPlaygroundResult(null); setEvalSummary(null);
  }

  function openChatForSource(source) {
    setSelectedSourceForChat(source);
    setTab("AI Paper Chat");
    setChatMessages([
      {
        role: "assistant",
        content: `👋 Ready to compare your paper with **"${source.title}"** (${source.similarity_percentage}% similarity).\n\nAsk me anything about their methodology, datasets, baseline models, key findings, or how to answer the challenge questions thrown by the review agents!`
      }
    ]);
  }

  async function sendChatMessage(customQuestion) {
    const q = customQuestion || chatInput;
    if (!q.trim() || chatBusy) return;
    const newMsgs = [...chatMessages, { role: "user", content: q }];
    setChatMessages(newMsgs);
    setChatInput("");
    setChatBusy(true);
    try {
      const context = JSON.stringify({
        paper_title: report?.paper_title || "Uploaded Research Paper",
        paper_text: "",
        selected_source: selectedSourceForChat,
        messages: chatMessages,
        question: q,
      });
      let res;
      if (file) {
        const form = new FormData();
        form.append("file", file);
        form.append("context", context);
        res = await fetch(API + "/api/chat/pdf", { method: "POST", body: form });
      } else {
        res = await fetch(API + "/api/chat", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: context,
        });
      }
      const data = await res.json();
      if (!res.ok) throw Error(data.detail || "Chat request failed");
      setChatMessages([...newMsgs, { role: "assistant", content: data.reply }]);
      if (data.suggested_questions?.length) {
        setSuggestedQuestions(data.suggested_questions);
      }
    } catch (err) {
      setChatMessages([...newMsgs, { role: "assistant", content: "⚠️ " + err.message }]);
    } finally {
      setChatBusy(false);
    }
  }

  async function executeMcpPlayground() {
    setPlaygroundBusy(true);
    try {
      let args = {};
      if (playgroundTool === "text_statistics") {
        args = { paper_text: sample };
      } else {
        args = { query: playgroundQuery, max_results: 5 };
      }
      const res = await fetch(API + "/api/mcp/invoke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: playgroundTool,
          arguments: args,
          calling_agent: "Interactive Presentation Console"
        })
      });
      const data = await res.json();
      if (!res.ok) throw Error(data.detail || "MCP invocation failed");
      setPlaygroundResult(data);
    } catch (err) {
      setPlaygroundResult({ error: err.message });
    } finally {
      setPlaygroundBusy(false);
    }
  }

  async function runEvaluationBenchmark() {
    setEvalBusy(true);
    setEvalStep("Starting the benchmark");
    try {
      setEvalSummary(await requestAnalysis(API + "/api/evaluation", {}, setEvalStep));
    } catch (err) {
      alert("Evaluation note: " + err.message);
    } finally {
      setEvalBusy(false);
      setEvalStep("");
    }
  }


  const findings = report?.findings || [];
  const agentActivities = report?.agent_activities || [];
  const challengeQuestions = report?.challenge_questions || [];
  const agentConversation = report?.agent_conversation || [];
  const guardrailAudits = report?.guardrail_audits || [];
  const mcpInvocations = report?.mcp_invocations || [];
  const stateTransitions = report?.state_transitions || [];
  const a2aExchanges = report?.a2a_exchanges || [];
  const modelUsage = report?.model_usage || [];
  const modelTokens = modelUsage.reduce((total, call) => total + call.input_tokens + call.output_tokens, 0);
  const modelCost = modelUsage.length && modelUsage.every(call => call.estimated_cost_usd != null)
    ? modelUsage.reduce((total, call) => total + call.estimated_cost_usd, 0)
    : null;

  const filteredDebate = agentConversation.filter(m => {
    if (debateFilter === "challenges") return !!m.challenge_question;
    if (debateFilter === "a2a") return m.challenge_question?.category === "adversarial_critique";
    return true;
  });

  useEffect(() => {
    function syncWorkflow() {
      setAppMode(workflowFromLocation());
      setSelectedPaperForViewer(null);
    }
    window.addEventListener("popstate", syncWorkflow);
    return () => window.removeEventListener("popstate", syncWorkflow);
  }, []);

  function navigate(workflow, href) {
    if (workflow === appMode) return;
    window.history.pushState(null, "", href);
    setSelectedPaperForViewer(null);
    setAppMode(workflow);
    window.scrollTo(0, 0);
  }

  return (
    <>
      <TubelightNavbar active={appMode} onNavigate={navigate} />
      {appMode === "idea" && !ideaReport ? (
      <main className="entry-screen">
        <div className="idea-entry">
          <form className="idea-search" onSubmit={event => { event.preventDefault(); submitIdea(); }}>
            <input
              aria-label="Describe your research idea"
              placeholder={ideaFile ? ideaFile.name : "Describe your research idea"}
              value={ideaFile ? "" : ideaText}
              readOnly={Boolean(ideaFile)}
              disabled={ideaBusy}
              onChange={event => setIdeaText(event.target.value)}
            />
            {ideaFile && <button type="button" className="idea-icon" aria-label="Remove attachment" disabled={ideaBusy} onClick={() => setIdeaFile(null)}><X size={18} /></button>}
            <Dropzone compact accept={proposalFiles} file={ideaFile} disabled={ideaBusy} onError={setIdeaError}
              onSelect={selected => { setIdeaFile(selected); setIdeaError(""); setIdeaSteps([]); }} />
            <button type="submit" className="idea-icon idea-submit" aria-label="Check novelty"
              disabled={ideaBusy || (!ideaFile && ideaText.trim().length < 20)}>
              {ideaBusy ? <LoaderCircle size={18} className="spin" /> : <Search size={18} />}
            </button>
          </form>
          <div className="idea-progress">
            <AnalysisProgress steps={ideaSteps} running={ideaBusy} failed={Boolean(ideaError)} />
            {ideaError && <p className="error" role="alert">{ideaError}</p>}
          </div>
        </div>
      </main>
      ) : (
      <main className="workflow-page">
        {appMode === "review" && (
        <div className={report ? "content" : "content review-entry"}>
          {!report && (
            <section className="review-upload" aria-labelledby="review-heading">
              <h1 id="review-heading">Paper Review</h1>
              <Dropzone accept={pdfFiles} file={file} disabled={busy || overviewBusy} onError={setError}
                onSelect={uploadPaper} />
              <AnalysisProgress steps={overviewSteps} running={overviewBusy} failed={Boolean(error && !overview)} />
              {overview && <PaperOverview overview={overview} />}

              {error && <div className="error"><X size={15} />{error}</div>}

              <AnalysisProgress steps={reviewSteps} running={busy} failed={Boolean(error)} />
              <div className="submitrow">
                <button
                  className="primary"
                  disabled={busy || overviewBusy || !file}
                  onClick={() => file && !overview ? uploadPaper(file) : submit()}
                >
                  {busy ? (
                    <><LoaderCircle className="spin" /> Reviewing paper…</>
                  ) : (
                    <>{file && !overview ? "Retry overview" : "Run paper review"} <ArrowUpRight size={15} /></>
                  )}
                </button>
              </div>
            </section>
          )}

          {/* Results Screen */}
          {report && (
            <section className="results">
              <div className="resulthead">
                <div>
                  <div className="eyebrow">— REVIEW COMPLETE · ID #{report.review_id}</div>
                  <h2>{report.paper_title}</h2>
                  <p>Generated multi-agent review · {new Date().toLocaleString()}</p>
                </div>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  <button className="ghost" onClick={() => setViewTableMode(!viewTableMode)}>
                    <Table size={14} /> {viewTableMode ? "Card View" : "Table View"}
                  </button>
                  <ReportDownloads report={report} api={API} />
                </div>
              </div>

              {(overview || report.overview) && <PaperOverview overview={overview || report.overview} />}

              <AnalysisProgress steps={reviewSteps} />

              {/* Metrics Header */}
              <div className="stats">
                <div>
                  <small>NOVELTY ESTIMATE</small>
                  <b>{report.novelty_score}<i>/100</i></b>
                  <span>Literature differentiation</span>
                </div>
                <div>
                  <small>FEASIBILITY</small>
                  <b>{report.feasibility_score}<i>/100</i></b>
                  <span>Methods &amp; hygiene audit</span>
                </div>
                <div>
                  <small>EVIDENCE COVERAGE</small>
                  <b>{Math.round(report.evidence_coverage * 100)}<i>%</i></b>
                  <span>{report.sources.length} multi-source citations</span>
                </div>
                <div>
                  <small>EXECUTION LATENCY</small>
                  <b>{(report.duration_ms / 1000).toFixed(1)}<i>s</i></b>
                  <span>{report.execution_mode.replaceAll("_", " ")}</span>
                </div>
              </div>

              {/* Recommendation Banner */}
              <div className="recommend">
                <ShieldCheck />
                <div>
                  <small>PANEL CONSENSUS RECOMMENDATION</small>
                  <b>{report.synthesis_recovered ? "Specialist results recovered" : report.overall_recommendation.replaceAll("_", " ")}</b>
                  <p>{report.summary}</p>
                </div>
                <Badge tone={report.overall_recommendation === "promising" ? "green" : report.overall_recommendation === "major_revision" ? "red" : "amber"}>
                  {report.synthesis_recovered ? "Specialist results recovered" : report.overall_recommendation.replaceAll("_", " ")}
                </Badge>
              </div>

              {/* Scoring Provenance & Real Process Audit Trail */}
              {report.scoring_provenance && (
                <div style={{
                  background: "rgba(16, 185, 129, 0.08)",
                  border: "1px solid rgba(16, 185, 129, 0.25)",
                  borderRadius: "8px",
                  padding: "10px 14px",
                  marginBottom: "16px",
                  fontSize: "12px",
                  color: "#d1d5db",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: "8px"
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{
                      background: "rgba(16, 185, 129, 0.2)",
                      color: "#34d399",
                      padding: "2px 8px",
                      borderRadius: "6px",
                      fontWeight: "600",
                      fontSize: "11px"
                    }}>
                      🛡️ LLM scoring
                    </span>
                    <span>
                      <b>Methodology:</b> {report.scoring_provenance.methodology}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: "12px", color: "#9ca3af", fontSize: "11px" }}>
                    <span><b>Similarity model:</b> {report.scoring_provenance.similarity_engine}</span>
                  </div>
                </div>
              )}

              <nav className="report-navigation" aria-label="Review sections">
                <div className="tabs">
                  {[
                    ["Findings", "Findings"],
                    ["Literature", "Literature & evidence"],
                    ["AI Paper Chat", "Paper chat"]
                  ].map(([value, label]) => (
                    <button key={value} className={tab === value ? "sel" : ""} aria-pressed={tab === value} onClick={() => setTab(value)}>
                      {label}
                    </button>
                  ))}
                </div>
                <details className="review-details">
                  <summary>Review details</summary>
                  <div className="detail-tabs">
                    {[
                      ["Agent Debate & Challenges", "Agent debate"],
                      ["Context & State Updates", "Context and state"],
                      ["MCP Tools & Protocol", "MCP tools"],
                      ["Guardrails & Security", "Guardrails"],
                      ["A2A Inter-Agent Protocol", "Agent-to-agent protocol"],
                      ["Evaluation & Cost", "Evaluation and cost"],
                      ["Agentic Activity", "Agent activity"],
                      ["Critique & logs", "Critique and logs"]
                    ].map(([value, label]) => (
                      <button key={value} className={tab === value ? "sel" : ""} aria-pressed={tab === value} onClick={() => setTab(value)}>
                        {label}
                      </button>
                    ))}
                  </div>
                </details>
              </nav>

              {/* TAB 1: FINDINGS (Structured Markdown Table or Cards - Req 3) */}
              {tab === "Findings" && (
                viewTableMode ? (
                  <div style={{ overflowX: "auto", background: "#f6f8fc", border: "1px solid #d4deed", borderRadius: "10px", padding: "16px" }}>
                    <h3 style={{ fontSize: "12px", color: "#17685e", marginTop: 0 }}>
                      Structured Findings Table (Pydantic Schema Output)
                    </h3>
                    <table className="eval-table">
                      <thead>
                        <tr>
                          <th>Dimension</th>
                          <th>Verdict</th>
                          <th>Headline</th>
                          <th>Confidence</th>
                          <th>Evidence Status</th>
                          <th>Revision Recommendation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {findings.map((f, i) => (
                          <tr key={i}>
                            <td style={{ fontWeight: 700, color: "#315879", textTransform: "capitalize" }}>{f.dimension.replace("_", " ")}</td>
                            <td><Badge tone={f.verdict === "positive" ? "green" : f.verdict === "mixed" ? "amber" : "red"}>{f.verdict}</Badge></td>
                            <td style={{ fontWeight: 600 }}>{f.headline}</td>
                            <td>{Math.round(f.confidence * 100)}%</td>
                            <td><Badge tone={f.evidence_status === "supported" ? "green" : "amber"}>{f.evidence_status.replace("_", " ")}</Badge></td>
                            <td style={{ color: "#526176" }}>{f.recommendation}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="cards">
                    {findings.map((f, i) => (
                      <article key={i} className="finding">
                        <div className="findicon">
                          {i === 0 ? <Search /> : i === 1 ? <FlaskConical /> : <Gauge />}
                        </div>
                        <div className="findbody">
                          <div className="findmeta">
                            <span>
                              {({
                                novelty: "Novelty & research gap",
                                methodology: "Methodology & feasibility",
                                dataset_evidence: "Dataset & evidence"
                              })[f.dimension] || f.dimension}
                            </span>
                            <Badge tone={f.evidence_status === "unverified" ? "red" : f.evidence_status === "partially_supported" ? "amber" : "green"}>
                              {f.evidence_status.replaceAll("_", " ")}
                            </Badge>
                          </div>
                          <h3>{f.headline}</h3>
                          <p>{f.explanation}</p>
                          <div className="next">
                            <b>Suggested next step for revision:</b>
                            <p>{f.recommendation}</p>
                          </div>
                          {f.source_ids?.length > 0 && (
                            <div className="pills">
                              {f.source_ids.map(s => <span key={s}>{s}</span>)}
                            </div>
                          )}
                        </div>
                        <div className="confidence">
                          {Math.round(f.confidence * 100)}%
                          <small>confidence</small>
                        </div>
                      </article>
                    ))}
                  </div>
                )
              )}

              {/* TAB 2: AGENT DEBATE & CHALLENGES (User Req 2) */}
              {tab === "Agent Debate & Challenges" && (
                <div className="debate-container">
                  <div className="debate-controls">
                    <div>
                      <b style={{ color: "#17685e", fontSize: "11px" }}>
                        Multi-Agent Peer Review Dialogue ({agentConversation.length} turns)
                      </b>
                      <p style={{ margin: "2px 0 0", fontSize: "10px", color: "#526176" }}>
                        Watch the specialist agents throw challenge questions at the author and debate findings in real time.
                      </p>
                    </div>
                    <div className="debate-filter-btns">
                      <button
                        className={"filter-btn " + (debateFilter === "all" ? "active" : "")}
                        onClick={() => setDebateFilter("all")}
                      >
                        All Turns ({agentConversation.length})
                      </button>
                      <button
                        className={"filter-btn " + (debateFilter === "challenges" ? "active" : "")}
                        onClick={() => setDebateFilter("challenges")}
                      >
                        ⚡ Challenge Questions Only ({challengeQuestions.length})
                      </button>
                      <button
                        className={"filter-btn " + (debateFilter === "a2a" ? "active" : "")}
                        onClick={() => setDebateFilter("a2a")}
                      >
                        🛡️ A2A Adversarial Pass
                      </button>
                    </div>
                  </div>

                  {filteredDebate.map((m, idx) => (
                    <div key={idx}>
                      {(idx === 0 || m.round !== filteredDebate[idx - 1]?.round) && (
                        <div className="debate-round-divider">
                          <span>{m.phase.toUpperCase()}</span>
                        </div>
                      )}

                      <div className="debate-turn">
                        <div className={"debate-avatar " + (m.avatar_icon || "workflow")}>
                          {m.avatar_icon === "workflow" ? <Workflow size={18} /> :
                           m.avatar_icon === "search" ? <Search size={18} /> :
                           m.avatar_icon === "microscope" ? <Microscope size={18} /> :
                           m.avatar_icon === "flask" ? <FlaskConical size={18} /> :
                           m.avatar_icon === "gauge" ? <Gauge size={18} /> :
                           m.avatar_icon === "shield" ? <ShieldCheck size={18} /> :
                           m.avatar_icon === "user-check" ? <UserCheck size={18} /> :
                           <Sparkles size={18} />}
                        </div>

                        <div className="debate-content">
                          <div className="debate-header">
                            <span className="debate-sender">{m.sender_name}</span>
                            <Badge tone={m.sender_framework.includes("CrewAI") ? "green" : m.sender_framework.includes("ADK") ? "amber" : ""}>
                              {m.sender_framework}
                            </Badge>
                            <small style={{ color: "#526176", fontSize: "9px", marginLeft: "auto" }}>
                              {m.timestamp}
                            </small>
                          </div>

                          <div className="debate-speech">{m.speech}</div>

                          {/* Highlighted Challenge Question Card */}
                          {m.challenge_question && (
                            <div className={"challenge-box " + (m.challenge_question.urgency === "critical" ? "critical" : "")}>
                              <div className="challenge-box-head">
                                <span className="challenge-tag">
                                  ⚡ CHALLENGE QUESTION THROWN
                                </span>
                                <Badge tone={m.challenge_question.urgency === "critical" ? "red" : "amber"}>
                                  {m.challenge_question.urgency.toUpperCase()} PRIORITY
                                </Badge>
                                <span className="challenge-target">
                                  Target: <b>{m.challenge_question.target}</b>
                                </span>
                              </div>
                              <p className="challenge-q-text">
                                "{m.challenge_question.question}"
                              </p>
                              <div className="challenge-context">
                                📌 Context reference: <i>{m.challenge_question.context_source}</i> · Status: <b style={{ color: "#17685e" }}>{m.challenge_question.status.replaceAll("_", " ")}</b>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 3: CONTEXT & STATE UPDATES (Req 1) */}
              {tab === "Context & State Updates" && (
                <div className="state-flow-list">
                  <div className="a2a-banner">
                    <h3>Context Sharing &amp; State Updates (Requirement 1)</h3>
                    <p>
                      PaperLens maintains an explicit shared state dictionary (<code>state = &#123;title, paper_text, stats, sources, findings, critique&#125;</code>)
                      that flows across pipeline stages. Intermediate outputs from upstream agents are deposited into state and feed directly into downstream specialists.
                    </p>
                  </div>

                  {stateTransitions.map((st, i) => (
                    <div key={i} className="state-flow-card">
                      <div className="state-step-num">{st.step_number}</div>
                      <div className="state-card-body">
                        <div className="state-card-head">
                          <b style={{ color: "#1c2939", fontSize: "12px" }}>{st.stage_name}</b>
                          <span className="state-handoff">
                            From: <b>{st.source_agent}</b> → To: <b>{st.target_agent}</b>
                          </span>
                          <small style={{ marginLeft: "auto", color: "#526176", fontSize: "9px" }}>{st.timestamp}</small>
                        </div>
                        <p className="state-summary">{st.intermediate_summary}</p>
                        <div className="state-keys">
                          <span style={{ fontSize: "9px", color: "#526176" }}>Shared State Keys Updated:</span>
                          {st.keys_updated.map(k => (
                            <span key={k} className="state-key-pill">state["{k}"]</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 4: MCP TOOLS & PROTOCOL (Req 2 & 4) */}
              {tab === "MCP Tools & Protocol" && (
                <div className="mcp-container">
                  <div className="mcp-arch-diagram">
                    <h3>Model Context Protocol (FastMCP) — Agent-to-Tool Interface (Requirement 2 &amp; 4)</h3>
                    <p>
                      All external tools are registered on the <b>FastMCP server</b> (<code>backend/mcp_server.py</code>)
                      and called over <b>Model Context Protocol JSON-RPC 2.0</b>. Agents never execute shell commands or direct HTTP calls.
                    </p>
                    <div className="mcp-flow">
                      <div className="mcp-flow-node active">
                        <Bot size={13} /> Specialist Agent
                      </div>
                      <span className="mcp-flow-arrow">→ tools/call JSON-RPC 2.0 →</span>
                      <div className="mcp-flow-node">
                        <Terminal size={13} /> MCP Client (app/mcp_client.py)
                      </div>
                      <span className="mcp-flow-arrow">→ stdio / FastMCP →</span>
                      <div className="mcp-flow-node active">
                        <Cpu size={13} /> FastMCP Server (mcp_server.py)
                      </div>
                      <span className="mcp-flow-arrow">→ query APIs →</span>
                      <div className="mcp-flow-node">
                        <BookOpen size={13} /> arXiv / OpenAlex / CrossRef
                      </div>
                    </div>
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <b style={{ color: "#17685e", fontSize: "11px" }}>
                      Executed MCP Tool Invocations during Review ({mcpInvocations.length} calls)
                    </b>
                    <small style={{ color: "#526176", fontSize: "9px" }}>
                      Transport: stdio / JSON-RPC 2.0 Spec
                    </small>
                  </div>

                  <div className="mcp-invocations-list">
                    {mcpInvocations.map((inv, idx) => (
                      <div key={idx} className="mcp-inv-card">
                        <div className="mcp-inv-head">
                          <Terminal size={14} style={{ color: "#17685e" }} />
                          <span className="mcp-tool-title">{inv.tool_name}()</span>
                          <Badge tone="green">{inv.protocol}</Badge>
                          <Badge tone={inv.status === "SUCCESS" ? "green" : "amber"}>{inv.status}</Badge>
                          <span style={{ fontSize: "9px", color: "#526176", marginLeft: "auto" }}>
                            Invoked by: <b style={{ color: "#315879" }}>{inv.agent_name}</b> · {inv.duration_ms} ms
                          </span>
                        </div>
                        <p style={{ margin: "4px 0 8px", fontSize: "10px", color: "#526176" }}>
                          {inv.response_summary}
                        </p>
                        <details style={{ marginTop: "6px" }}>
                          <summary style={{ cursor: "pointer", fontSize: "9px", color: "#17685e", fontWeight: 600 }}>
                            Inspect JSON-RPC 2.0 Protocol Envelope (Request &amp; Response)
                          </summary>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginTop: "8px" }}>
                            <div>
                              <small style={{ color: "#526176", fontSize: "8px" }}>MCP REQUEST (JSON-RPC 2.0)</small>
                              <pre className="json-box">{JSON.stringify(inv.request_jsonrpc, null, 2)}</pre>
                            </div>
                            <div>
                              <small style={{ color: "#526176", fontSize: "8px" }}>MCP RESPONSE (JSON-RPC 2.0)</small>
                              <pre className="json-box">{JSON.stringify(inv.response_jsonrpc, null, 2)}</pre>
                            </div>
                          </div>
                        </details>
                      </div>
                    ))}
                  </div>

                  {/* Interactive MCP Tool Playground */}
                  <div className="playground-box">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Code2 size={16} style={{ color: "#17685e" }} />
                      <h3 style={{ margin: 0 }}>Interactive MCP Tool Console (Live Presentation Demo)</h3>
                      <Badge tone="green">Interactive Tester</Badge>
                    </div>
                    <p style={{ margin: 0, fontSize: "10px", color: "#526176" }}>
                      Select any registered MCP tool and invoke it live to demonstrate how agents trigger tools over JSON-RPC:
                    </p>
                    <div className="playground-row">
                      <select value={playgroundTool} onChange={e => setPlaygroundTool(e.target.value)}>
                        <option value="search_academic_literature">search_academic_literature (Federated arXiv/OpenAlex/CrossRef)</option>
                        <option value="search_arxiv">search_arxiv (arXiv Preprints)</option>
                        <option value="search_openalex">search_openalex (OpenAlex 250M Catalog)</option>
                        <option value="search_crossref">search_crossref (CrossRef DOI Database)</option>
                        <option value="text_statistics">text_statistics (Structural Paper Audit)</option>
                      </select>
                      {playgroundTool !== "text_statistics" && (
                        <input
                          value={playgroundQuery}
                          onChange={e => setPlaygroundQuery(e.target.value)}
                          placeholder="Academic search query..."
                        />
                      )}
                      <button
                        className="primary"
                        onClick={executeMcpPlayground}
                        disabled={playgroundBusy}
                        style={{ padding: "8px 14px" }}
                      >
                        {playgroundBusy ? <LoaderCircle className="spin" size={12} /> : <Play size={12} />} Execute via MCP
                      </button>
                    </div>

                    {playgroundResult && (
                      <div>
                        <small style={{ color: "#17685e", fontSize: "9px", fontWeight: 700 }}>
                          LIVE MCP EXECUTION RESULT:
                        </small>
                        <pre className="json-box" style={{ maxHeight: "220px" }}>
                          {JSON.stringify(playgroundResult, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* TAB 5: GUARDRAILS & SECURITY (Req 8) */}
              {tab === "Guardrails & Security" && (
                <div className="guardrails-container">
                  <div className="guardrails-banner">
                    <div>
                      <small>GUARDRAIL ARCHITECTURE</small>
                      <b>4 Active Layers</b>
                      <span>Defense-in-depth</span>
                    </div>
                    <div>
                      <small>LAYER 1 AUDIT</small>
                      <b>Input Ingestion</b>
                      <span>Injection, toxicity, size</span>
                    </div>
                    <div>
                      <small>LAYER 2 AUDIT</small>
                      <b>MCP Boundary</b>
                      <span>Type-check callbacks</span>
                    </div>
                    <div>
                      <small>LAYER 3 &amp; 4 AUDIT</small>
                      <b>A2A &amp; Schema</b>
                      <span>Sandboxing &amp; Pydantic</span>
                    </div>
                  </div>

                  {[
                    {
                      layer: "Layer 1: Input Ingestion",
                      title: "Layer 1 — Input Ingestion Guardrails",
                      subtitle: "Enforced before any LLM sees input: prompt injection scanner, toxic/abusive filter, character boundaries (backend/app/guardrails.py)",
                      icon: <ShieldAlert size={16} style={{ color: "#17685e" }} />
                    },
                    {
                      layer: "Layer 2: Tool & MCP Boundary",
                      title: "Layer 2 — Tool Execution & Return Type-Checking Callbacks",
                      subtitle: "Enforced at the FastMCP tool boundary: query sanitization, tool return type checking, and safe output truncation",
                      icon: <Terminal size={16} style={{ color: "#8a540e" }} />
                    },
                    {
                      layer: "Layer 3: Multi-Agent & A2A Boundary",
                      title: "Layer 3 — Multi-Agent & A2A Context Guardrails",
                      subtitle: "Sandboxes untrusted text in XML tags and verifies citation grounding against hallucination",
                      icon: <Workflow size={16} style={{ color: "#315879" }} />
                    },
                    {
                      layer: "Layer 4: Output Synthesis & Schema",
                      title: "Layer 4 — Output Synthesis & Schema Guardrails",
                      subtitle: "Score boundary clamping [0, 100], logical verdict consistency, and strict Pydantic schema validation",
                      icon: <CheckCircle size={16} style={{ color: "#17685e" }} />
                    }
                  ].map(sec => {
                    const secAudits = guardrailAudits.filter(a => a.layer === sec.layer);
                    return (
                      <div key={sec.layer} className="layer-section">
                        <div className="layer-head">
                          {sec.icon}
                          <div>
                            <h3>{sec.title}</h3>
                            <p>{sec.subtitle}</p>
                          </div>
                        </div>

                        {secAudits.map((a, i) => (
                          <div key={i} className="guardrail-card">
                            <div className="guardrail-top">
                              <span className="guardrail-name">{a.guardrail_name}</span>
                              <span className="code-pill">{a.code_location}</span>
                              <Badge tone={a.status === "PASSED" ? "green" : a.status === "WARNING" ? "amber" : "red"}>
                                {a.status}
                              </Badge>
                            </div>
                            <div className="guardrail-desc">
                              <b>Policy:</b> {a.policy_description}
                            </div>
                            <div className="guardrail-finding">
                              <b>Audit Evidence:</b> {a.audit_findings}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* TAB 6: A2A INTER-AGENT PROTOCOL (Req 6 & 7) */}
              {tab === "A2A Inter-Agent Protocol" && (
                <div className="a2a-container">
                  <div className="a2a-banner">
                    <h3>A2A Inter-Agent Communication Protocol (Requirement 6 &amp; 7)</h3>
                    <p>
                      PaperLens uses <b>CrewAI and Google ADK</b>. The CrewAI specialist agents
                      evaluate Novelty, Methodology, and Evidence, then package findings into a standardized JSON-RPC 2.0 message
                      sent across the <b>A2A Protocol boundary</b> (<code>/.well-known/agent-card.json</code> and <code>/a2a</code>)
                      to the Google ADK critic. Its model calls use the configured provider fallback chain.
                    </p>
                  </div>

                  <div className="a2a-handshake-bar">
                    <Workflow size={14} style={{ color: "#644991" }} />
                    <span>A2A Handshake:</span>
                    <Badge tone="green">/.well-known/agent-card.json</Badge>
                    <span style={{ color: "#17685e" }}>Sender: CrewAI Specialists</span>
                    <span>→ JSON-RPC 2.0 →</span>
                    <span style={{ color: "#8a540e" }}>Receiver: Google ADK Critic</span>
                  </div>

                  {a2aExchanges.map((ex, i) => (
                    <div key={i} className="a2a-record-card">
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                        <b style={{ color: "#1c2939", fontSize: "12px" }}>A2A Message Exchange #{i + 1}</b>
                        <Badge tone="green">{ex.protocol}</Badge>
                        <Badge tone="amber">{ex.status}</Badge>
                        <small style={{ marginLeft: "auto", color: "#526176", fontSize: "9px" }}>{ex.duration_ms} ms</small>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginTop: "10px" }}>
                        <div>
                          <small style={{ color: "#526176", fontSize: "8px" }}>A2A REQUEST (CrewAI → ADK)</small>
                          <pre className="json-box">{JSON.stringify(ex.request_payload, null, 2)}</pre>
                        </div>
                        <div>
                          <small style={{ color: "#526176", fontSize: "8px" }}>A2A RESPONSE (ADK → CrewAI)</small>
                          <pre className="json-box">{JSON.stringify(ex.response_payload, null, 2)}</pre>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 7: EVALUATION & COST (Req 10) */}
              {tab === "Evaluation & Cost" && (
                <div className="eval-container">
                  <div className="a2a-banner">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
                      <div>
                        <h3>End-to-End Performance, Running Time &amp; Cost Benchmark (Requirement 10)</h3>
                        <p>Evaluates multi-agent peer review latency, running time in seconds, and token cost across diverse scientific domains.</p>
                      </div>
                      <button className="primary" onClick={runEvaluationBenchmark} disabled={evalBusy}>
                        {evalBusy ? <><LoaderCircle className="spin" size={12} /> Running Benchmark...</> : <><Play size={12} /> Run Benchmark Suite</>}
                      </button>
                    </div>
                  </div>
                  {evalStep && <p className="request-status" role="status">{evalStep}</p>}

                  {/* Current Paper Run Stats */}
                  <div className="eval-grid">
                    <div className="eval-stat">
                      <small>THIS PAPER LATENCY</small>
                      <b>{(report.duration_ms / 1000).toFixed(2)}s</b>
                      <span>{report.duration_ms.toFixed(0)} ms running time</span>
                    </div>
                    <div className="eval-stat">
                      <small>MODEL TOKENS</small>
                      <b>{modelUsage.length ? modelTokens.toLocaleString() : "Unavailable"}</b>
                      <span>{modelUsage.length} model responses · {modelUsage.filter(call => call.estimated).length} estimated</span>
                    </div>
                    <div className="eval-stat">
                      <small>MODEL API COST</small>
                      <b>{modelCost == null ? "Unavailable" : `$${modelCost.toFixed(6)}`}</b>
                      <span>Estimated provider charges</span>
                    </div>
                    <div className="eval-stat">
                      <small>GUARDRAIL INTEGRITY</small>
                      <b>{guardrailAudits.filter(g => g.status === "PASSED").length}/{guardrailAudits.length}</b>
                      <span>Security checks passed</span>
                    </div>
                  </div>

                  {/* Benchmark Suite Summary if run */}
                  {evalSummary && (
                    <div style={{ background: "#f6f8fc", border: "1px solid #d4deed", borderRadius: "10px", padding: "16px" }}>
                      <h3 style={{ fontSize: "12px", color: "#17685e", marginTop: 0 }}>
                        Benchmark Results ({evalSummary.total_cases_evaluated} Academic Test Cases)
                      </h3>
                      <div className="eval-grid" style={{ marginBottom: "14px" }}>
                        <div className="eval-stat">
                          <small>AVERAGE RUNNING TIME</small>
                          <b>{evalSummary.average_running_time_s}s</b>
                          <span>Mean latency {evalSummary.average_latency_ms} ms</span>
                        </div>
                        <div className="eval-stat">
                          <small>TOTAL SUITE COST</small>
                          <b>{evalSummary.total_estimated_cost_usd == null ? "Unavailable" : `$${evalSummary.total_estimated_cost_usd.toFixed(6)}`}</b>
                          <span>USD across all cases</span>
                        </div>
                        <div className="eval-stat">
                          <small>COST PER REVIEW</small>
                          <b>{evalSummary.average_cost_per_review_usd == null ? "Unavailable" : `$${evalSummary.average_cost_per_review_usd.toFixed(6)}`}</b>
                          <span>USD / research paper</span>
                        </div>
                        <div className="eval-stat">
                          <small>TOTAL EVALUATED</small>
                          <b>{evalSummary.passed_cases}/{evalSummary.total_cases_evaluated} passed</b>
                          <span>End-to-end checks</span>
                        </div>
                      </div>

                      <table className="eval-table">
                        <thead>
                          <tr>
                            <th>Case ID</th>
                            <th>Paper Title</th>
                            <th>Running Time</th>
                            <th>Novelty</th>
                            <th>Feasibility</th>
                            <th>Tokens</th>
                            <th>Est. Cost</th>
                            <th>Guardrails</th>
                            <th>Result</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evalSummary.results.map((r, i) => (
                            <tr key={i}>
                              <td style={{ fontWeight: 700, color: "#17685e" }}>{r.id}</td>
                              <td>{r.title}</td>
                              <td>{r.running_time_seconds}s</td>
                              <td>{r.novelty_score == null ? "—" : `${r.novelty_score}/100`}</td>
                              <td>{r.feasibility_score == null ? "—" : `${r.feasibility_score}/100`}</td>
                              <td>{r.estimated_tokens.toLocaleString()}</td>
                              <td>{r.estimated_cost_usd == null ? "Unavailable" : `$${r.estimated_cost_usd.toFixed(6)}`}</td>
                              <td><Badge tone="green">{r.guardrails_passed} Passed</Badge></td>
                              <td>{r.passed ? "Pass" : r.issues.join("; ")}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 8: LITERATURE */}
              {tab === "Literature" && (
                report.sources.length ? (
                  <div className="cards">
                    <p className="request-status">Similarity estimates use the titles and abstracts available from the literature indexes.</p>
                    {report.sources.map((s, i) => (
                      <article key={i} className="source">
                        <BookOpen />
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px", flexWrap: "wrap" }}>
                            <h3>{s.title}</h3>
                            <EvidenceBadge paper={s} />
                            <Badge tone={i === 0 ? "green" : s.similarity_percentage >= 65 ? "amber" : ""}>
                              {i === 0 ? "TOP MATCH · " : ""}{s.similarity_percentage}% SIMILARITY
                            </Badge>
                            <Badge tone={s.source === "arXiv" ? "green" : s.source === "OpenAlex" ? "amber" : ""}>
                              {s.source || "Academic"}
                            </Badge>
                          </div>
                          <p>{s.authors?.slice(0, 3).join(", ")} {s.year ? `(${s.year})` : ""}</p>
                          <div style={{ display: "flex", gap: "8px", alignItems: "center", marginTop: "10px", flexWrap: "wrap" }}>
                            <button className="ghost" onClick={() => setSelectedPaperForViewer({
                              ...s, overlap_summary: s.relevance_explanation,
                            })}>
                              Inspect evidence
                            </button>
                            <a href={s.url} target="_blank" rel="noreferrer" className="ghost" style={{ padding: "6px 10px", fontSize: "9px" }}>
                              Open source ↗
                            </a>
                            <button
                              className="ghost"
                              style={{ padding: "6px 10px", fontSize: "9px", color: "#17685e", borderColor: "#cce2dd" }}
                              onClick={() => openChatForSource(s)}
                            >
                              <MessageSquare size={12} /> Discuss with AI Chatbot
                            </button>
                          </div>
                          {s.relevance_explanation && <p>{s.relevance_explanation}</p>}
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="empty">No sources retrieved. Check MCP and network access.</p>
                )
              )}

              {/* TAB 9: AGENTIC ACTIVITY */}
              {tab === "Agentic Activity" && (
                <div className="cards">
                  {agentActivities.map((a, i) => (
                    <article key={i} className="finding">
                      <div className="findicon">
                        {a.agent_id === "orchestrator" ? <Workflow /> :
                         a.agent_id === "literature_retriever" ? <Search /> :
                         a.agent_id === "novelty_analyst" ? <Microscope /> :
                         a.agent_id === "methodology_reviewer" ? <FlaskConical /> :
                         a.agent_id === "dataset_evidence_reviewer" ? <Gauge /> :
                         a.agent_id === "adversarial_critic" ? <ShieldCheck /> :
                         <Sparkles />}
                      </div>
                      <div className="findbody">
                        <div className="findmeta">
                          <span>{a.name}</span>
                          <Badge tone={a.framework.includes("CrewAI") ? "green" : a.framework.includes("MCP") ? "amber" : a.framework.includes("ADK") ? "green" : ""}>
                            {a.framework}
                          </Badge>
                          <Badge tone={a.status === "completed" ? "green" : "amber"}>{a.status}</Badge>
                          {a.duration_ms != null && (
                            <small style={{ color: "#526176", fontSize: "9px", marginLeft: "auto" }}>
                              {a.duration_ms} ms
                            </small>
                          )}
                        </div>
                        <h3 style={{ marginTop: "3px" }}>{a.action}</h3>
                        <div className="next">
                          <b>Internal reasoning &amp; audit trail:</b>
                          <p>{a.reasoning}</p>
                        </div>
                        <p style={{ marginTop: "8px", color: "#526176" }}>
                          <b>Agent decision:</b> {a.output_summary}
                        </p>
                        {a.metrics && Object.keys(a.metrics).length > 0 && (
                          <div className="pills">
                            {Object.entries(a.metrics).map(([k, v]) => (
                              <span key={k}>{k.replaceAll("_", " ")}: <b>{String(v)}</b></span>
                            ))}
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              )}

              {/* TAB 10: CRITIQUE & LOGS */}
              {tab === "Critique & logs" && (
                <div className="logs">
                  <div className="critique">
                    <b><ShieldCheck /> Adversarial critique from Google ADK</b>
                    <p>{report.critique_summary}</p>
                  </div>
                  <h3>Pipeline Execution Trace · {report.events.length} events logged</h3>
                  {report.langfuse_trace_id && <p>Langfuse trace ID <code>{report.langfuse_trace_id}</code></p>}
                  {report.events.map((e, i) => (
                    <div key={i} className="event">
                      <span className={"eventdot " + e.status} />
                      <div>
                        <b>{e.stage}</b>
                        <p>{e.message}</p>
                      </div>
                      <small>{e.duration_ms != null ? e.duration_ms + " ms" : e.status}</small>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 11: AI PAPER CHAT */}
              {tab === "AI Paper Chat" && (
                <div className="chat-window">
                  <div className="source-select-bar">
                    <BookOpen size={14} style={{ color: "#17685e" }} />
                    <span>Focus Paper:</span>
                    <select
                      value={selectedSourceForChat?.source_id || ""}
                      onChange={e => {
                        const found = report.sources.find(s => s.source_id === e.target.value);
                        setSelectedSourceForChat(found || null);
                      }}
                    >
                      <option value="">All Fetched Literature (General Overview)</option>
                      {report.sources.map(s => (
                        <option key={s.source_id} value={s.source_id}>
                          {s.similarity_percentage}% Match · {s.title.slice(0, 55)}...
                        </option>
                      ))}
                    </select>
                    {selectedSourceForChat && (
                      <Badge tone="green">{selectedSourceForChat.similarity_percentage}% Similarity</Badge>
                    )}
                  </div>

                  <div className="quick-prompts">
                    {suggestedQuestions.map((q, idx) => (
                      <button key={idx} className="quick-btn" onClick={() => sendChatMessage(q)}>
                        {q}
                      </button>
                    ))}
                  </div>

                  <div className="chat-history">
                    {chatMessages.length === 0 && (
                      <div style={{ color: "#526176", fontSize: "11px", lineHeight: 1.7, padding: "10px" }}>
                        <p style={{ color: "#17685e", fontWeight: 700, marginBottom: "6px" }}>
                          <Sparkles size={13} style={{ display: "inline", marginRight: 4 }} /> PaperLens AI Research Copilot
                        </p>
                        Ask any question about how your uploaded paper compares with the fetched literature, how to respond to the agents' challenge questions, or how the Guardrails, MCP tools, and A2A interoperability work in this system.
                      </div>
                    )}
                    {chatMessages.map((m, idx) => (
                      <div key={idx} className={"chat-msg " + m.role}>
                        <div className="chat-bubble">
                          <div style={{ fontWeight: 700, fontSize: "9px", marginBottom: "4px", color: m.role === "user" ? "#70dfca" : "#9eb2d0" }}>
                            {m.role === "user" ? "YOU" : "PAPERLENS COPILOT"}
                          </div>
                          <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
                        </div>
                      </div>
                    ))}
                    {chatBusy && (
                      <div className="chat-msg assistant">
                        <div className="chat-bubble">
                          <LoaderCircle className="spin" size={13} /> Generating research response...
                        </div>
                      </div>
                    )}
                  </div>

                  <form className="chat-input-row" onSubmit={e => { e.preventDefault(); sendChatMessage(); }}>
                    <input
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      placeholder="Ask anything about challenges, guardrails, MCP tools, methodology..."
                      disabled={chatBusy}
                    />
                    <button className="primary" type="submit" disabled={chatBusy || !chatInput.trim()}>
                      <Send size={13} /> Send
                    </button>
                  </form>
                </div>
              )}

              {/* Limitations Notice */}
              <div className="caution">
                <b><ShieldCheck size={15} /> Read this before relying on the review</b>
                <ul>
                  {report.limitations.map((x, i) => <li key={i}>{x}</li>)}
                </ul>
              </div>

              <button className="primary" onClick={reset}>
                Review another paper <ArrowUpRight size={15} />
              </button>
            </section>
          )}

        </div>
        )}

        {/* Feature: Check Research Idea Section */}
        {appMode === "idea" && (
          <div className="content">
            {/* Research report */}
            {ideaReport && (
              <div className="idea-container">
                <div className="resulthead">
                  <div>
                    <div className="eyebrow">Research idea report</div>
                    <h1>{ideaReport.title}</h1>
                    <p>{ideaReport.total_papers_searched} papers searched</p>
                    {ideaReport.langfuse_trace_id && <p>Langfuse trace ID <code>{ideaReport.langfuse_trace_id}</code></p>}
                  </div>
                  <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    <ReportDownloads report={ideaReport} api={API} />
                    <button className="ghost" onClick={resetIdea}>
                      <RefreshCw size={14} /> Check another idea
                    </button>
                  </div>
                </div>

                <AnalysisProgress steps={ideaSteps} />
                <ResearchTree report={ideaReport} onInspect={setSelectedPaperForViewer} />
              </div>
            )}

          </div>
        )}

        {selectedPaperForViewer && (
          <PaperViewer paper={selectedPaperForViewer} api={API} onClose={() => setSelectedPaperForViewer(null)} />
        )}
      </main>
      )}
    </>
  );
}
