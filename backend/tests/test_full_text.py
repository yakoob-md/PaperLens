from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from app import full_text


class FullTextTests(TestCase):
    def test_supported_urls_canonicalize_without_forwarding_untrusted_targets(self):
        self.assertEqual(full_text.supported_pdf("http://export.arxiv.org/abs/2209.15001v3"),
                         "https://arxiv.org/pdf/2209.15001v3")
        self.assertEqual(full_text.supported_pdf("https://aclanthology.org/2024.acl-long.1/"),
                         "https://aclanthology.org/2024.acl-long.1.pdf")
        self.assertIsNone(full_text.supported_pdf("https://arxiv.org@example.com/pdf/2209.15001"))
        self.assertEqual(full_text.supported_pdf("https://proceedings.mlr.press/v119/chen20j.html"),
                         "https://proceedings.mlr.press/v119/chen20j/chen20j.pdf")


class EvidenceTests(IsolatedAsyncioTestCase):
    async def test_abstract_only_source_preserves_exact_located_passage(self):
        quote = "Mulching increased moisture retention compared with untreated plots."
        source = {"url": "https://example.org/paper", "title": "Water retention", "abstract": quote}
        await full_text.check_full_text([source], "Does mulching increase moisture retention?")
        evidence = source["full_text_evidence"]
        self.assertEqual(evidence.status, "abstract_only")
        self.assertEqual(evidence.passages, [])
        self.assertEqual(evidence.abstract_passages[0].quote, quote)
        self.assertIn("retention", evidence.abstract_passages[0].shared_terms)

    async def test_evidence_quotes_are_verified_without_altering_abstract_scores(self):
        quote = "The mulched plots retained more water than the untreated plots after six weeks."
        sources = [
            *[{"url": "https://example.org/article", "title": "Unavailable"} for _ in range(3)],
            {"url": "https://arxiv.org/abs/2209.15001", "title": "Match", "similarity_percentage": 64},
        ]
        result = full_text._EvidenceResult(passages=[
            full_text._EvidenceItem(passage_id="1", comparison="Same outcome measure."),
            full_text._EvidenceItem(passage_id="missing", comparison="Unsupported."),
            full_text._EvidenceItem(passage_id="1", comparison="Duplicate."),
        ])
        with (
            patch.object(full_text, "fetch_pdf", AsyncMock(return_value=b"pdf")),
            patch.object(full_text, "_passages", return_value=[{"page": 4, "passage_id": "1", "text": quote}]),
            patch.object(full_text, "complete", AsyncMock(return_value=result)),
        ):
            await full_text.check_full_text(sources, "Mulching")
        verified = sources[3]["full_text_evidence"]
        self.assertEqual(verified.status, "full_text_evidence")
        self.assertEqual([item.page for item in verified.passages], [4])
        self.assertEqual(verified.passages[0].quote, quote)
        self.assertEqual(sources[3]["similarity_percentage"], 64)
        self.assertEqual(sources[1]["full_text_evidence"].status, "abstract_only")
