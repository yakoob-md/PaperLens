from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app import crew, llm
from app.config import Settings
from app.schemas import PaperOverview


class LocalModelTests(IsolatedAsyncioTestCase):
    def setUp(self):
        config = patch.object(llm, "settings", Settings(
            _env_file=None, ollama_model="test-surface", ollama_agent_model="test-agent",
        ))
        config.start()
        self.addCleanup(config.stop)

    @staticmethod
    def reply(content):
        return httpx.Response(200, json={"message": {"content": content}},
                              request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"))

    async def test_structured_retry_stays_on_ollama_and_preserves_schema(self):
        valid = self.reply(PaperOverview(about="Trial", key_outcomes=[], search_query="soil moisture").model_dump_json())
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=[self.reply("{}"), valid])) as post:
            with llm.model_session():
                result = await llm.complete("Summarize", {"paper_id": "ID"}, PaperOverview)
                self.assertEqual(result.about, "Trial")
                self.assertEqual(llm.preferred_model()[1], "test-surface")
        self.assertEqual(post.await_count, 2)
        first = post.await_args_list[0]
        self.assertEqual(first.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(first.kwargs["json"]["format"], PaperOverview.model_json_schema())

    async def test_agentic_role_uses_separate_model(self):
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=self.reply("Critique"))) as post:
            self.assertEqual(await llm.complete("Review", {}, agentic=True), "Critique")
        self.assertEqual(post.await_args.kwargs["json"]["model"], "test-agent")

    def test_streamed_reply_requires_completion_and_joins_chunks(self):
        headers = {"content-type": "application/x-ndjson"}
        partial = '{"message":{"content":"Local "},"done":false}\n'
        final = '{"message":{"content":"review"},"done":true}\n'
        self.assertEqual(llm._reply_text(httpx.Response(200, headers=headers, text=partial + final)), "Local review")
        with self.assertRaisesRegex(ValueError, "before completion"):
            llm._reply_text(httpx.Response(200, headers=headers, text=partial))

    async def test_connection_error_reports_local_service(self):
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("offline"))):
            with self.assertRaisesRegex(llm.ModelUnavailable, "Start the Ollama service"):
                await llm.complete("Review", {})


class GeminiModelTests(IsolatedAsyncioTestCase):
    async def test_three_specialist_reviews_share_one_cloud_call(self):
        native = Mock(acall=AsyncMock(return_value=crew.SpecialistReviews(
            novelty="NOVELTY_SCORE: 47",
            methodology="FEASIBILITY_SCORE: 70",
            evidence="EVIDENCE_COVERAGE: 66",
        )))
        settings = Settings(_env_file=None, google_api_key="test-key", gemini_model="test-gemini")
        with (
            patch.object(llm, "settings", settings),
            patch.object(crew, "settings", settings),
            patch("crewai.LLM", return_value=native) as factory,
        ):
            result = await crew.run_crewai_review("Test paper", "A synthetic trial.", [])
        self.assertEqual(result["model"], "test-gemini")
        self.assertEqual(len(result["tasks"]), 3)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["model"], "gemini/test-gemini")
        native.acall.assert_awaited_once()

    async def test_nested_schema_uses_gemini_json_and_tracks_model(self):
        from pydantic import create_model

        schema = create_model("Review", specialists=(crew.SpecialistReviews, ...))
        review = schema(specialists=crew.SpecialistReviews(novelty="New", methodology="Sound", evidence="Limited"))
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
                replies = [LocalModelTests.reply("Local review")]
                if status:
                    cloud = httpx.Request("POST", llm.GEMINI_URL)
                    replies.insert(0, httpx.Response(status, request=cloud))
                with (
                    patch.object(llm, "settings", Settings(_env_file=None, google_api_key=key)),
                    patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=replies)) as post,
                    llm.model_session(),
                    self.assertLogs(llm.logger, "WARNING") as logs,
                ):
                    self.assertEqual(await llm.complete("Review", {}, provider="gemini"), "Local review")
                    self.assertEqual(llm.preferred_model()[0], "ollama")
                self.assertEqual(post.await_count, len(replies))
                self.assertTrue(post.await_args.args[0].endswith("/api/chat"))
                self.assertIn("Falling back to Ollama", logs.output[0])
