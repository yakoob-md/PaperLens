from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import json
import httpx

from app import crew, llm
from app.config import Settings
from app.schemas import PaperOverview


class GroqModelTests(IsolatedAsyncioTestCase):
    def setUp(self):
        config = patch.object(llm, "settings", Settings(
            _env_file=None, groq_api_key="test-key", groq_model="test-surface", groq_agent_model="test-agent",
        ))
        config.start()
        self.addCleanup(config.stop)

    @staticmethod
    def reply(content, finish_reason="stop"):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                       "usage": {"prompt_tokens": 20, "completion_tokens": 8}},
            request=httpx.Request("POST", f"{llm.GROQ_URL}/chat/completions"),
        )

    async def test_structured_retry_preserves_schema_and_caps_prompt(self):
        valid = self.reply(PaperOverview(about="Trial", key_outcomes=[], search_query="soil moisture").model_dump_json())
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=[self.reply("{}"), valid])) as post:
            with llm.model_session():
                result = await llm.complete("Summarize", {"paper_id": "ID", "text": "?" * 50000}, PaperOverview)
                self.assertEqual(result.about, "Trial")
                self.assertEqual(llm.preferred_model()[:2], ("groq", "test-surface"))
        self.assertEqual(post.await_count, 2)
        for call in post.await_args_list:
            self.assertEqual(call.args[0], f"{llm.GROQ_URL}/chat/completions")
            payload = call.kwargs["json"]
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertIn("paper_id", payload["messages"][1]["content"])
            self.assertLessEqual(sum(len(message["content"].encode("utf-8")) for message in payload["messages"]), 5000)

    async def test_agentic_model_retries_truncation_with_bounded_budget(self):
        config = llm.settings.model_copy(update={"groq_output_tokens": 500})
        with (
            patch.object(llm, "settings", config),
            patch.object(httpx.AsyncClient, "post", AsyncMock(
                side_effect=[self.reply("", "length"), self.reply("Critique")],
            )) as post,
        ):
            result = await llm.complete("Review", {}, agentic=True, max_output_tokens=320)
        self.assertEqual(result, "Critique")
        self.assertEqual([call.kwargs["json"]["max_completion_tokens"] for call in post.await_args_list], [320, 500])
        self.assertEqual(post.await_args.kwargs["json"]["model"], "test-agent")

    def test_crowded_synthesis_keeps_all_source_ids(self):
        from app.pipeline import LocalReviewSynthesis

        sources = [{"source_id": f"source-{i}", "title": "T" * 160, "abstract": "A" * 150}
                   for i in range(7)]
        evidence = {
            "title": "T" * 240, "overview": "O" * 850, "key_outcomes": ["K" * 220] * 4,
            "paper_excerpts": "P" * 3600, "sources": sources, "specialist_notes": ["S" * 850] * 3,
            "document_statistics": {"word_count": 240, "sentence_count": 20},
        }
        messages = llm.groq_messages("Synthesize a concise peer review. " * 15, evidence, LocalReviewSynthesis)
        compacted = json.loads(messages[-1]["content"])
        self.assertEqual([source["source_id"] for source in compacted["sources"]],
                         [source["source_id"] for source in sources])

    async def test_rate_limit_waits_for_groq_before_retrying(self):
        limited = httpx.Response(429, headers={"retry-after": "2"},
                                 request=httpx.Request("POST", llm.GROQ_URL))
        with (
            patch.object(llm.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=[limited, self.reply("Review")])),
        ):
            self.assertEqual(await llm.complete("Review", {}), "Review")
        sleep.assert_awaited_once_with(2)

    async def test_long_rate_limit_reports_quota_without_retrying(self):
        limited = httpx.Response(429, headers={"retry-after": "3600"},
                                 request=httpx.Request("POST", llm.GROQ_URL))
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=limited)) as post:
            with self.assertRaisesRegex(llm.ModelUnavailable, "Groq rate limit reached"):
                await llm.complete("Review", {})
        post.assert_awaited_once()


class GeminiModelTests(IsolatedAsyncioTestCase):
    async def test_three_crewai_specialists_call_gemini_and_emit_callbacks(self):
        from app.observability import RunMonitor, current_monitor

        async def review(_instructions, messages, **options):
            self.assertEqual(options["provider"], "gemini")
            self.assertIn("A synthetic trial", str(messages))
            return "Final Answer: Specialist score: 66."

        monitor = RunMonitor()
        token = current_monitor.set(monitor)
        try:
            with patch.object(crew, "complete", AsyncMock(side_effect=review)) as completion:
                result = await crew.run_crewai_review("Test paper", "A synthetic trial.", [])
        finally:
            current_monitor.reset(token)
        self.assertEqual(len(result["tasks"]), 3)
        self.assertEqual(completion.await_count, 3)
        self.assertEqual(len([event for event in monitor.events if event.stage.startswith("callback.")]), 3)

    async def test_nested_schema_uses_gemini_json_and_tracks_model(self):
        from pydantic import BaseModel, create_model

        class SpecialistReviews(BaseModel):
            novelty: str
            methodology: str
            evidence: str

        schema = create_model("Review", specialists=(SpecialistReviews, ...))
        review = schema(specialists=SpecialistReviews(novelty="New", methodology="Sound", evidence="Limited"))
        url = f"{llm.GEMINI_URL}/test-gemini:generateContent"
        response = httpx.Response(200, request=httpx.Request("POST", url), json={
            "candidates": [{"content": {"parts": [{"text": review.model_dump_json()}]}, "finishReason": "STOP"}],
        })
        with (
            patch.object(llm, "settings", Settings(_env_file=None, google_api_key="test-key", gemini_model="test-gemini")),
            patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)) as post,
            llm.model_session(),
        ):
            self.assertEqual(await llm.complete("Review", {}, schema, provider="gemini"), review)
            self.assertEqual(llm.preferred_model()[:2], ("gemini", "test-gemini"))
        post.assert_awaited_once()
        self.assertEqual(post.await_args.args[0], url)
        config = post.await_args.kwargs["json"]["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseJsonSchema"], schema.model_json_schema())

    async def test_missing_invalid_and_quota_keys_fall_back(self):
        for key, status in (("", None), ("invalid", 400), ("forbidden", 403), ("exhausted", 429)):
            with self.subTest(status=status):
                replies = [GroqModelTests.reply("Groq review")]
                if status:
                    cloud = httpx.Request("POST", llm.GEMINI_URL)
                    replies.insert(0, httpx.Response(status, request=cloud))
                with (
                    patch.object(llm, "settings", Settings(_env_file=None, google_api_key=key, groq_api_key="test-key")),
                    patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=replies)) as post,
                    llm.model_session(),
                    self.assertLogs(llm.logger, "WARNING") as logs,
                ):
                    self.assertEqual(await llm.complete("Review", {}, provider="gemini"), "Groq review")
                    self.assertEqual(llm.preferred_model()[0], "groq")
                self.assertEqual(post.await_count, len(replies))
                self.assertTrue(post.await_args.args[0].endswith("/chat/completions"))
                self.assertIn("Falling back to Groq", logs.output[0])
