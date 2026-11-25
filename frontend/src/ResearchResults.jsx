
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  ArrowUpRight, BookOpen, ChevronRight, Compass, FileText, Highlighter,
  Layers3, Lightbulb, Target, X
} from "lucide-react";

function Branch({ label, Icon, meta, initiallyOpen, children }) {
  return (
    <details className="research-branch" open={initiallyOpen}>
      <summary>
        {Icon && <span className="branch-icon"><Icon size={17} aria-hidden="true" /></span>}
        <span className="branch-label">{label}</span>
        {meta && <span className="branch-meta">{meta}</span>}
        <ChevronRight className="branch-chevron" size={17} aria-hidden="true" />
      </summary>
      <div className="branch-content">{children}</div>
    </details>
  );
}

function ResearchNote({ label, value }) {
  if (!value?.length) return null;
  return (
    <Branch label={label}>
      {Array.isArray(value)
        ? <ul>{value.map((item, index) => <li key={index}>{item}</li>)}</ul>
        : <p>{value}</p>}
    </Branch>
  );
}

export function EvidenceBadge({ paper }) {
  const count = paper.full_text_evidence?.passages?.length ?? 0;
  return <span className="badge">{count ? `${count} PDF ${count === 1 ? "highlight" : "highlights"}` : "Abstract only"}</span>;
}

export function FullTextEvidence({ paper }) {
  const evidence = paper.full_text_evidence;
  return (
    <section className="viewer-evidence" aria-label="Full-text evidence">
      <h3>Full-text evidence</h3>
      <p>{evidence?.note || "This source has an abstract-based comparison."}</p>
      {evidence?.passages.map((item, index) => (
        <figure key={index}>
          <figcaption>
            {evidence.pdf_url
              ? <a href={`${evidence.pdf_url.split("#")[0]}#page=${item.page}`} target="_blank" rel="noopener noreferrer">
                  Open PDF page {item.page} <ArrowUpRight size={13} aria-hidden="true" />
                </a>
              : `PDF page ${item.page}`}
          </figcaption>
          <blockquote><mark>{item.quote}</mark></blockquote>
          <p>{item.comparison}</p>
        </figure>
      ))}
    </section>
  );
}

export function ResearchTree({ report, onInspect }) {
  const { analysis, assessment, research_gaps: gaps, recommendations, similar_papers: papers } = report;
  return (
    <div className="research-tree" aria-label="Research idea report">
      <Branch label="Idea overview" Icon={Compass}>
        <ResearchNote label="Research problem" value={analysis.problem_statement} />
        <ResearchNote label="Objectives" value={analysis.research_objectives} />
        <ResearchNote label="Methods" value={analysis.proposed_methodology} />
        <ResearchNote label="Research domain" value={analysis.research_domain} />
        <ResearchNote label="Key concepts and technologies" value={analysis.key_concepts_and_technologies} />
        <ResearchNote label="Potential contributions" value={analysis.potential_contributions} />
        <ResearchNote label="Keywords" value={analysis.important_keywords} />
      </Branch>
      <Branch label="Novelty assessment" Icon={Target}>
        <div className="novelty-summary">
          {assessment.novelty_score != null && <strong>{assessment.novelty_score}<small> / 100</small></strong>}
          <div>
            <h3>{assessment.category}</h3>
            <p>{assessment.explanation}</p>
          </div>
        </div>
        <ResearchNote label="Observations" value={assessment.observations} />
        {report.scoring_provenance && (
          <ResearchNote label="Scoring provenance & methodology" value={[
            `Methodology: ${report.scoring_provenance.methodology}`,
            `Similarity model: ${report.scoring_provenance.similarity_engine}`,
            `Novelty model: ${report.scoring_provenance.novelty_auditor}`,
            ...(report.scoring_provenance.audit_trail || [])
          ]} />
        )}
        <ResearchNote label="Already explored" value={assessment.explored_aspects} />
        <ResearchNote label="Similar aspects" value={assessment.similar_aspects} />
        <ResearchNote label="Ways to differentiate" value={assessment.distinctive_opportunities} />
        <ResearchNote label="Assessment limitations" value={assessment.disclaimer} />
      </Branch>
      <Branch label="Related papers and evidence" Icon={BookOpen} meta={String(papers.length)} initiallyOpen>
        {papers.length > 0 && <p>Overlap estimates use the titles and available abstracts.</p>}
        {papers.length === 0 && <p>No related papers were retrieved.</p>}
        {papers.map(paper => (
          <div key={paper.paper_id}>
          <Branch
            label={paper.title}
            meta={paper.similarity_percentage == null ? "Unscored" : `${paper.similarity_percentage}% overlap`}
          >
            <p className="paper-metadata">
              {[paper.authors.join(", "), paper.year, paper.venue, paper.doi && "DOI " + paper.doi].filter(Boolean).join(" · ")}
            </p>
            <div className="research-actions">
              <EvidenceBadge paper={paper} />
              <a className="ghost" href={paper.url} target="_blank" rel="noopener noreferrer">
                Open source <ArrowUpRight size={15} />
              </a>
            </div>
            <FullTextEvidence paper={paper} />
            <ResearchNote label="Abstract" value={paper.abstract} />
            <ResearchNote label={"Overlap summary · " + paper.similarity_percentage + "% similarity"} value={paper.overlap_summary} />
          </Branch>
          <button type="button" className="ghost" aria-label={`Inspect evidence for ${paper.title}`} onClick={() => onInspect(paper)}>
            <Highlighter size={15} /> Inspect evidence
          </button>
          </div>
        ))}
      </Branch>
      <Branch label="Research gaps" Icon={Layers3}>
        {!Object.values(gaps).some(items => items.length) && <p>The available records do not establish a research gap.</p>}
        <ResearchNote label="Covered areas" value={gaps.covered_areas} />
        <ResearchNote label="Underexplored areas" value={gaps.underexplored_areas} />
        <ResearchNote label="Research directions" value={gaps.suggested_directions} />
        <ResearchNote label="Differentiation opportunities" value={gaps.differentiation_opportunities} />
      </Branch>
      <Branch label="Recommendations" Icon={Lightbulb} meta={String(recommendations.length)}>
        {recommendations.length === 0 && <p>No recommendations were generated.</p>}
        {recommendations.map((recommendation, index) => {
          const match = recommendation.match(/^\*\*(.*?)\*\*:?\s*([\s\S]*)$/);
          return (
            <Branch key={index} label={match?.[1] || "Recommendation " + (index + 1)}>
              <p>{match?.[2] || recommendation}</p>
            </Branch>
          );
        })}
      </Branch>
    </div>
  );
}

function originalPdf(paper) {
  for (const link of [paper.pdf_url, paper.url]) {
    if (!link) continue;
    try {
      const url = new URL(link);
      if (url.protocol !== "https:" && url.protocol !== "http:") continue;
      if (["arxiv.org", "www.arxiv.org", "export.arxiv.org"].includes(url.hostname)) {
        const match = url.pathname.match(/^\/(?:abs|pdf)\/(.+?)(?:\.pdf)?$/);
        if (match) return { url: `https://arxiv.org/pdf/${match[1]}`, arxiv: true };
      }
      if (link === paper.pdf_url || /\.pdf$/i.test(url.pathname)) return { url: url.href, arxiv: false };
    } catch {
      // Ignore malformed registry metadata.
    }
  }
  return null;
}

function OriginalPdfPanel({ pdf, api }) {
  const [preview, setPreview] = useState(null);
  const [failed, setFailed] = useState(false);
  const arxivUrl = pdf?.arxiv ? pdf.url : null;

  useEffect(() => {
    if (!arxivUrl) return;
    const controller = new AbortController();
    let blobUrl;
    async function load() {
      try {
        const response = await fetch(api + "/api/idea/proxy-pdf?url=" + encodeURIComponent(arxivUrl), {
          signal: controller.signal,
        });
        if (!response.ok || !response.headers.get("content-type")?.includes("application/pdf")) {
          throw new Error("Preview unavailable");
        }
        const blob = await response.blob();
        if (!controller.signal.aborted) {
          blobUrl = URL.createObjectURL(blob);
          setPreview(blobUrl);
        }
      } catch {
        if (!controller.signal.aborted) setFailed(true);
      }
    }
    load();
    return () => {
      controller.abort();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [api, arxivUrl]);

  if (!pdf) return <p className="viewer-instruction">A direct PDF was not provided by the source. Use the publisher page to check full-text access.</p>;
  if (failed) return <p className="viewer-instruction" role="status">The inline preview is unavailable. Open the PDF in a new tab.</p>;
  if (arxivUrl && !preview) return <p className="viewer-instruction" role="status">Loading the original PDF...</p>;
  return <iframe title="Original academic paper PDF" src={preview || pdf.url} />;
}

export function PaperViewer({ paper, api, onClose }) {
  const dialog = useRef(null);
  const detailClose = useRef(null);
  const passageButtons = useRef(new Map());
  const [view, setView] = useState(() => paper.full_text_evidence?.passages?.length ? "evidence" : "passages");
  const [pdfOpened, setPdfOpened] = useState(false);
  const [selection, setSelection] = useState(null);
  const [category, setCategory] = useState("");
  const highlights = paper.highlights || [];
  const abstractPassages = paper.full_text_evidence?.abstract_passages || [];
  const visibleHighlights = highlights.filter(highlight => !category || highlight.category === category);
  const active = visibleHighlights.find(highlight => highlight.highlight_id === selection);
  const evidencePdf = paper.full_text_evidence?.pdf_url;
  const pdf = originalPdf(evidencePdf ? { ...paper, pdf_url: evidencePdf } : paper);
  const source = pdf?.url || paper.url;

  useLayoutEffect(() => {
    const element = dialog.current;
    const trigger = document.activeElement;
    element.showModal();
    return () => {
      element.close();
      trigger?.focus();
    };
  }, []);

  useLayoutEffect(() => {
    if (active) detailClose.current?.focus({ preventScroll: true });
  }, [active]);

  function closeDetails() {
    passageButtons.current.get(selection)?.focus();
    setSelection(null);
  }

  function jumpTo(event) {
    passageButtons.current.get(event.target.value)?.focus();
    event.target.value = "";
  }

  return (
    <dialog
      className="research-viewer"
      ref={dialog}
      aria-labelledby="paper-viewer-title"
      onCancel={event => { event.preventDefault(); onClose(); }}
      onClick={event => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div className="research-viewer-surface">
        <header className="viewer-heading">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span className={`badge ${paper.similarity_percentage === 100 ? "danger" : "info"}`} style={{
                background: paper.similarity_percentage === 100 ? "rgba(239, 68, 68, 0.2)" : "rgba(59, 130, 246, 0.2)",
                color: paper.similarity_percentage === 100 ? "#f87171" : "#60a5fa",
                border: `1px solid ${paper.similarity_percentage === 100 ? "rgba(239, 68, 68, 0.4)" : "rgba(59, 130, 246, 0.4)"}`,
                padding: "2px 8px",
                borderRadius: "9999px",
                fontSize: "11px",
                fontWeight: "600"
              }}>
                {paper.similarity_percentage == null ? "Unscored source" : `${paper.similarity_percentage}% Conceptual Overlap`}
              </span>
            </div>
            <h2 id="paper-viewer-title">{paper.title}</h2>
            <p>{[paper.venue, paper.year].filter(Boolean).join(" · ")}</p>
          </div>
          <button type="button" className="icon-button" aria-label="Close paper viewer" onClick={onClose}>
            <X size={18} />
          </button>
        </header>
        <div className="viewer-toolbar">
          <div className="viewer-switch" aria-label="Paper view">
            <button type="button" aria-pressed={view === "passages"} onClick={() => setView("passages")}>
              <Highlighter size={15} /> Abstract comparison
            </button>
            <button type="button" aria-pressed={view === "evidence"} onClick={() => setView("evidence")}>
              <BookOpen size={15} /> Full-text evidence
            </button>
            <button type="button" aria-pressed={view === "pdf"} onClick={() => {
              setSelection(null);
              setPdfOpened(true);
              setView("pdf");
            }}>
              <FileText size={15} /> Original PDF
            </button>
          </div>
          {view === "passages" && highlights.length > 1 && (
            <div className="viewer-filters">
              <select aria-label="Filter by overlap category" value={category} onChange={event => {
                setSelection(null);
                setCategory(event.target.value);
              }}>
                <option value="">All categories</option>
                {[...new Set(highlights.map(highlight => highlight.category))].map(name =>
                  <option key={name} value={name}>{name}</option>
                )}
              </select>
              {visibleHighlights.length > 1 && (
                <select aria-label="Jump to a passage" defaultValue="" onChange={jumpTo}>
                  <option disabled value="">Jump to passage</option>
                  {visibleHighlights.map((highlight, index) => (
                    <option key={highlight.highlight_id} value={highlight.highlight_id}>
                      {index + 1}. {highlight.category}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}
          {view === "pdf" && (
            <a className="viewer-source" href={source} target="_blank" rel="noopener noreferrer">
              {pdf ? "Open PDF in new tab" : "Open publisher page"} <ArrowUpRight size={14} />
            </a>
          )}
        </div>
        <div className="viewer-content" hidden={view !== "passages"}>
          <div className="viewer-reader">
            <article className="viewer-paper">
              <h3>{highlights.length ? "Highlighted passages" : "Overlap assessment"}</h3>
              {paper.overlap_summary && <p>{paper.overlap_summary}</p>}
              <p className="viewer-instruction">
                {visibleHighlights.length ? "Select a passage to see why it overlaps." :
                  abstractPassages.length ? "These quotes are from the source abstract. Shared terms indicate possible overlap." :
                  paper.abstract ? "No specific overlap passage was identified. The source abstract is shown below." :
                  "The literature index supplied no abstract for this source."}
              </p>
              {!highlights.length && abstractPassages.map(({ quote, shared_terms }) => (
                <figure key={quote}>
                  <figcaption>Source abstract</figcaption>
                  <blockquote><mark>{quote}</mark></blockquote>
                  <p>Shared terms · {shared_terms.join(", ")}</p>
                </figure>
              ))}
              {visibleHighlights.map(highlight => (
                <figure key={highlight.highlight_id}>
                  <figcaption>{highlight.section}{highlight.page != null && (" · Page " + highlight.page)}</figcaption>
                  <button
                    ref={element => {
                      if (element) passageButtons.current.set(highlight.highlight_id, element);
                      else passageButtons.current.delete(highlight.highlight_id);
                    }}
                    type="button"
                    className={"passage-highlight " + highlight.severity}
                    aria-expanded={active?.highlight_id === highlight.highlight_id}
                    aria-controls={active?.highlight_id === highlight.highlight_id ? "overlap-detail" : undefined}
                    onClick={() => setSelection(active?.highlight_id === highlight.highlight_id ? null : highlight.highlight_id)}
                  >
                    {highlight.text_snippet}
                  </button>
                </figure>
              ))}
              {paper.abstract && (
                <details className="viewer-abstract" open={!highlights.length && !abstractPassages.length}>
                  <summary>Abstract</summary>
                  <p>{paper.abstract}</p>
                </details>
              )}
            </article>
          </div>
          {active && (
            <aside id="overlap-detail" className="viewer-insight" aria-labelledby="overlap-title">
              <div className="viewer-insight-heading">
                <h3 id="overlap-title">Overlap details</h3>
                <button ref={detailClose} type="button" className="icon-button" aria-label="Close overlap details" onClick={closeDetails}>
                  <X size={16} />
                </button>
              </div>
              <p className="insight-meta">{active.page != null ? ("Page " + active.page) : active.section}</p>
              <dl>
                <dt>Matching idea</dt><dd>{active.matched_idea_aspect}</dd>
                <dt>Why it overlaps</dt><dd>{active.explanation}</dd>
                <dt>How to differentiate</dt><dd>{active.differentiation_strategy}</dd>
              </dl>
            </aside>
          )}
        </div>
        {view === "evidence" && <FullTextEvidence paper={paper} />}
        {pdfOpened && (
          <div className="viewer-pdf" hidden={view !== "pdf"}>
            <OriginalPdfPanel pdf={pdf} api={api} />
          </div>
        )}
      </div>
    </dialog>
  );
}
