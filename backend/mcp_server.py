"""MCP server with multi-source academic research tools (arXiv, OpenAlex, CrossRef)."""
import re, urllib.parse, urllib.request, xml.etree.ElementTree as ET, json
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("paperlens-research-tools")

@mcp.tool()
def text_statistics(paper_text: str) -> dict:
    """Count words/sentences and detect common paper sections."""
    text_sample = paper_text[:80000]
    words = re.findall(r"\b[\w'-]+\b", text_sample)
    sentences = re.findall(r"[.!?]+", text_sample)
    return {
        "characters": len(paper_text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_abstract": bool(re.search(r"\babstract\b", text_sample, re.I)),
        "has_methodology": bool(re.search(r"\b(method|methodology|approach|architecture)\b", text_sample, re.I)),
        "has_experiments": bool(re.search(r"\b(experiment|evaluation|results|benchmark)\b", text_sample, re.I)),
        "has_limitations": bool(re.search(r"\b(limitation|future work|ethical)\b", text_sample, re.I)),
        "has_references": bool(re.search(r"\b(references|bibliography)\b", text_sample, re.I)),
    }

def _reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        positions = []
        for word, idx_list in inverted_index.items():
            for idx in idx_list:
                positions.append((idx, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)[:1500]
    except Exception:
        return ""

@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict:
    """Search public arXiv preprint metadata and abstracts."""
    n = max(1, min(int(max_results), 8))
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"all:{query[:160]}", "start": 0, "max_results": n,
        "sortBy": "relevance", "sortOrder": "descending"})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperLens/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read(1000000)
        root = ET.fromstring(raw)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for e in root.findall("a:entry", ns):
            papers.append({
                "source_id": f"arxiv-{e.findtext('a:id', default='', namespaces=ns).split('/')[-1]}",
                "title": " ".join(e.findtext("a:title", default="", namespaces=ns).split()),
                "authors": [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)],
                "published": e.findtext("a:published", default="", namespaces=ns)[:4],
                "url": e.findtext("a:id", default="", namespaces=ns),
                "abstract": " ".join(e.findtext("a:summary", default="", namespaces=ns).split())[:1500],
                "source": "arXiv"
            })
        return {"papers": papers, "count": len(papers)}
    except Exception as exc:
        return {"papers": [], "count": 0, "error": f"arXiv error: {type(exc).__name__}"}

@mcp.tool()
def search_openalex(query: str, max_results: int = 5) -> dict:
    """Search OpenAlex open catalog of 250M+ scientific articles and conference proceedings."""
    n = max(1, min(int(max_results), 8))
    clean_q = urllib.parse.quote(query[:160])
    url = f"https://api.openalex.org/works?search={clean_q}&per-page={n}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperLens/1.0 (mailto:academic-research@paperlens.local)"})
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read(1500000)
        data = json.loads(raw)
        papers = []
        for r_item in data.get("results", []):
            title = r_item.get("title")
            if not title:
                continue
            authors = [a.get("author", {}).get("display_name", "") for a in r_item.get("authorships", [])[:5] if a.get("author")]
            year = r_item.get("publication_year")
            url_link = r_item.get("doi") or r_item.get("id") or "https://openalex.org"
            abstract = _reconstruct_openalex_abstract(r_item.get("abstract_inverted_index"))
            oa_id = str(r_item.get("id", "")).split("/")[-1] or str(len(papers) + 1)
            papers.append({
                "source_id": f"openalex-{oa_id}",
                "title": " ".join(title.split()),
                "authors": [a for a in authors if a],
                "published": str(year) if year else None,
                "url": url_link,
                "abstract": abstract or f"OpenAlex catalog indexed work published in {year or 'recent years'}.",
                "source": "OpenAlex"
            })
        return {"papers": papers, "count": len(papers)}
    except Exception as exc:
        return {"papers": [], "count": 0, "error": f"OpenAlex error: {type(exc).__name__}"}

@mcp.tool()
def search_crossref(query: str, max_results: int = 5) -> dict:
    """Search CrossRef catalog of 150M+ published DOI articles, IEEE, ACM, Springer, Elsevier."""
    n = max(1, min(int(max_results), 8))
    clean_q = urllib.parse.quote(query[:160])
    url = f"https://api.crossref.org/works?query={clean_q}&rows={n}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperLens/1.0 (mailto:academic-research@paperlens.local)"})
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read(1500000)
        data = json.loads(raw)
        papers = []
        for item in data.get("message", {}).get("items", []):
            titles = item.get("title") or []
            if not titles or not titles[0]:
                continue
            title = titles[0]
            authors = [f"{a.get('given','')} {a.get('family','')}".strip() for a in item.get("author", [])[:5]]
            date_info = (item.get("published-print") or item.get("published-online") or item.get("created") or {})
            date_parts = date_info.get("date-parts", [[None]])
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            doi = item.get("DOI") or ""
            url_link = item.get("URL") or (f"https://doi.org/{doi}" if doi else "https://crossref.org")
            abstract = re.sub(r"<[^>]+>", "", item.get("abstract", ""))[:1500]
            cr_id = doi.replace("/", "-") if doi else str(len(papers) + 1)
            papers.append({
                "source_id": f"crossref-{cr_id[:24]}",
                "title": " ".join(title.split()),
                "authors": [a for a in authors if a],
                "published": str(year) if year else None,
                "url": url_link,
                "abstract": abstract or f"Published scholarly publication in {item.get('container-title', ['academic proceedings'])[0] if item.get('container-title') else 'peer-reviewed literature'}.",
                "source": "CrossRef"
            })
        return {"papers": papers, "count": len(papers)}
    except Exception as exc:
        return {"papers": [], "count": 0, "error": f"CrossRef error: {type(exc).__name__}"}

@mcp.tool()
def search_academic_literature(query: str, max_results: int = 8) -> dict:
    """Federated multi-source search querying arXiv, OpenAlex, and CrossRef simultaneously."""
    total_papers = []
    seen_titles = set()
    breakdown = {"arXiv": 0, "OpenAlex": 0, "CrossRef": 0}

    # Query arXiv
    try:
        arxiv_res = search_arxiv(query=query, max_results=4)
        for p in arxiv_res.get("papers", []):
            norm_title = "".join(re.findall(r"[a-z0-9]", p["title"].lower()))
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                total_papers.append(p)
                breakdown["arXiv"] += 1
    except Exception:
        pass

    # Query OpenAlex
    try:
        oa_res = search_openalex(query=query, max_results=4)
        for p in oa_res.get("papers", []):
            norm_title = "".join(re.findall(r"[a-z0-9]", p["title"].lower()))
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                total_papers.append(p)
                breakdown["OpenAlex"] += 1
    except Exception:
        pass

    # Query CrossRef
    try:
        cr_res = search_crossref(query=query, max_results=4)
        for p in cr_res.get("papers", []):
            norm_title = "".join(re.findall(r"[a-z0-9]", p["title"].lower()))
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                total_papers.append(p)
                breakdown["CrossRef"] += 1
    except Exception:
        pass

    return {
        "papers": total_papers[:max_results],
        "count": len(total_papers[:max_results]),
        "breakdown": breakdown
    }

if __name__ == "__main__":
    mcp.run()
