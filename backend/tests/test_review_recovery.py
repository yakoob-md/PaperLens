import json
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from app import a2a, pipeline
from app.llm import ModelUnavailable
from app.schemas import McpToolInvocation, PaperOverview


class ReviewRecoveryTests(IsolatedAsyncioTestCase):
    async def test_report_keeps_specialist_scores_when_late_steps_fail(self):
        text = "Mulched plots retained 12 percent more moisture than untreated plots. " * 1000
        notes = [
            "**NOVELTY_SCORE: 47/100**\nCompare prior work.\nCHALLENGE: What differentiates this trial?",
            "FEASIBILITY_SCORE: 70\nDesign is reproducible.",
            "EVIDENCE_COVERAGE: 66%\nUncertainty was not reported.",
        ]

        async def call_tool(name, *_args):
            return {}, McpToolInvocation(
                call_id=name, timestamp="", agent_name="", tool_name=name,
                response_summary="", duration_ms=0, status="SUCCESS",
            )

        overview = PaperOverview(
            about="A mulch trial", key_outcomes=["12 percent more moisture"], search_query="mulch moisture",
        )
        with (
            patch.object(pipeline, "summarize_paper", AsyncMock(return_value=overview)),
            patch.object(pipeline, "call_mcp_tool", call_tool),
            patch.object(pipeline, "run_crewai_review", AsyncMock(return_value={
                "tasks": [{"raw": note} for note in notes],
            })) as crew,
            patch.object(pipeline, "complete", AsyncMock(side_effect=ModelUnavailable("offline"))) as synthesis,
            patch.object(a2a, "execute_a2a_critique", AsyncMock(side_effect=ModelUnavailable("offline"))),
        ):
            report = await pipeline.run_review("Mulch study", text)

        self.assertEqual(crew.await_args.args[1], text)
        synthesis_input = synthesis.await_args.args[1]
        self.assertNotIn("paper_text", synthesis_input)
        self.assertLess(len(json.dumps(synthesis_input)), 10_000)
        self.assertTrue(report.synthesis_recovered)
        self.assertEqual((report.novelty_score, report.feasibility_score, report.evidence_coverage), (47, 70, .66))
        self.assertIsNone(report.findings[0].confidence)
        self.assertEqual(len(report.challenge_questions), 1)
        self.assertEqual(report.a2a_exchanges, [])

    def test_missing_or_invalid_specialist_scores_are_never_invented(self):
        notes = ["NOVELTY_SCORE: 47", "FEASIBILITY_SCORE: 101", "EVIDENCE_COVERAGE: 65"]
        self.assertIsNone(pipeline.recover_specialist_review(notes, [], "Overview"))
        self.assertIsNone(pipeline.recover_specialist_review(notes[:2], [], "Overview"))
