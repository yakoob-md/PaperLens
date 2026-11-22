import io
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from docx import Document
import httpx

from app import idea_checker

SYNOPSIS = """PaperLens project synopsis

Problem statement: Scientific peer review needs evidence-grounded feedback on submitted manuscripts.

Objectives: Retrieve related publications and verify citation grounding.

Methods: An agentic peer review workflow combines literature retrieval, specialist critique, and evidence validation.

Keywords: scientific peer review, citation grounding, literature retrieval
"""
RELATED = {
    "paper_id": "related", "title": "Scientific peer review with citation grounding",
    "authors": ["A. Researcher"], "year": 2025, "url": "https://example.org/paper",
    "doi": "10.0000/example", "abstract": "We evaluate citation grounding for scientific peer review using literature retrieval.",
}
ROBOTICS = {
    **RELATED, "paper_id": "robot", "title": "Multi-agent robotic navigation",
    "doi": "10.0000/robot", "abstract": "Agents learn a policy for coordinating robotic navigation.",
}


class ExtractionTests(TestCase):
    def test_docx_preserves_text_and_table_order(self):
        document = Document()
        document.add_heading("PaperLens project synopsis")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Methods"
        table.cell(0, 1).text = "Scientific peer review\nwith citation grounding"
        document.add_paragraph("Keywords: literature retrieval")
        data = io.BytesIO()
        document.save(data)
        text = idea_checker.extract_text_from_file(data.getvalue(), "synopsis.DOCX")
        self.assertIn("Methods\n\nScientific peer review with citation grounding", text)
        self.assertLess(text.index("citation grounding"), text.index("Keywords"))

    def test_pdf_extracts_later_pages(self):
        # Build a two-page text PDF without requiring a PDF generation dependency.
        first = b"BT /F1 12 Tf 40 700 Td (PaperLens synopsis) Tj ET"
        second = b"BT /F1 12 Tf 40 700 Td (Methods: citation grounding) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            *(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream" for content in (first, second)),
        ]
        pdf = b"%PDF-1.4\n"
        offsets = []
        for number, content in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += b"%d 0 obj\n" % number + content + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 8\n0000000000 65535 f \n"
        pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
        pdf += b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
        self.assertEqual(
            idea_checker.extract_text_from_file(pdf, "synopsis.pdf"),
            "PaperLens synopsis\n\nMethods: citation grounding",
        )

class PipelineTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.analysis = idea_checker.IdeaAnalysis(
            problem_statement="Scientific peer review needs evidence-grounded feedback.",
            research_objectives=["Retrieve related publications"],
            proposed_methodology="Literature retrieval and citation grounding",
            key_concepts_and_technologies=["citation grounding"],
            research_domain="Scientific peer review",
            potential_contributions=[],
            important_keywords=["scientific peer review", "citation grounding"],
        )

    def comparison(self, papers, score):
        return idea_checker.IdeaComparison(
            assessment=idea_checker.NoveltyAssessment(
                novelty_score=score, explanation="Evidence reviewed.", observations=[],
                explored_aspects=[], similar_aspects=[], distinctive_opportunities=[],
                disclaimer="Assessment is limited to the retrieved records.",
            ),
            papers=papers,
            research_gaps=idea_checker.ResearchGap(
                covered_areas=[], underexplored_areas=[], suggested_directions=[],
                differentiation_opportunities=[],
            ),
            recommendations=[],
        )

    async def test_model_receives_submitted_text_and_no_evidence_preserves_null_score(self):
        with patch.object(idea_checker, "complete", AsyncMock(side_effect=[self.analysis, self.comparison([], None)])) as model, \
             patch.object(idea_checker, "_fetch_openalex", AsyncMock(return_value=[])), \
             patch.object(idea_checker, "_fetch_arxiv", AsyncMock(side_effect=httpx.ConnectError("offline"))), \
             patch.object(idea_checker, "_fetch_crossref", AsyncMock(return_value=[])):
            report = await idea_checker.check_research_idea_novelty("PaperLens", SYNOPSIS)
        self.assertEqual(model.await_args_list[0].args[1]["document_text"], SYNOPSIS)
        self.assertEqual(report.similar_papers, [])
        self.assertIsNone(report.assessment.novelty_score)
        self.assertIn("Indexes unavailable: arXiv.", report.assessment.observations)

    async def test_model_scores_are_preserved_and_highlights_are_grounded(self):
        def highlight(snippet):
            return idea_checker.OverlapHighlight(
                highlight_id=snippet, text_snippet=snippet, matched_idea_aspect="citation grounding",
                explanation="Overlap in methodology", differentiation_strategy="Compare evaluation protocols.",
            )
        match = idea_checker.IdeaPaperAssessment(
            paper_id="related", similarity_percentage=73, relevance_category="high",
            overlap_summary="Both study citation grounding.",
            highlights=[highlight(RELATED["abstract"]), highlight("Invented quote")],
        )
        duplicate = {**RELATED, "paper_id": "duplicate", "doi": None, "abstract": ""}
        self.assertEqual(len(idea_checker._deduplicate([ROBOTICS, duplicate, RELATED])), 2)
        with patch.object(idea_checker, "complete", AsyncMock(side_effect=[self.analysis, self.comparison([match], 38)])), \
             patch.object(idea_checker, "_search", AsyncMock(return_value=([RELATED, ROBOTICS], ["OpenAlex"], []))):
            report = await idea_checker.check_research_idea_novelty("PaperLens", SYNOPSIS)
        self.assertEqual(report.assessment.novelty_score, 38)
        self.assertEqual(report.similar_papers[0].similarity_percentage, 73)
        self.assertEqual([item.text_snippet for item in report.similar_papers[0].highlights], [RELATED["abstract"]])
        self.assertIsNone(report.similar_papers[0].highlights[0].page)

    async def test_model_failure_is_propagated_before_literature_search(self):
        with patch.object(idea_checker, "complete", AsyncMock(side_effect=idea_checker.ModelUnavailable("offline"))), \
             patch.object(idea_checker, "_search", AsyncMock()) as search:
            with self.assertRaises(idea_checker.ModelUnavailable):
                await idea_checker.check_research_idea_novelty("PaperLens", SYNOPSIS)
        search.assert_not_awaited()
