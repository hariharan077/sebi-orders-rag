from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.news_lookup import CurrentNewsLookupProvider
from app.sebi_orders_rag.current_info.official_lookup import OfficialWebsiteCurrentInfoProvider
from app.sebi_orders_rag.directory_data.models import FetchedDirectorySource
from app.sebi_orders_rag.web_fallback.ranking import rank_web_sources
from app.sebi_orders_rag.web_fallback.models import WebSearchRequest
from app.sebi_orders_rag.web_fallback.models import WebSearchResult, WebSearchSource
from app.sebi_orders_rag.web_fallback.provider import OpenAIResponsesWebSearchProvider


class OfficialSearchPriorityTests(unittest.TestCase):
    def test_current_lookup_uses_official_web_after_structured_miss(self) -> None:
        provider = OfficialWebsiteCurrentInfoProvider(
            settings=_settings(),
            fetcher=_EmptyStructuredFetcher(),
            official_search_provider=_FakeOfficialWebProvider(
                answer_text=(
                    "SEBI states that its income comes primarily from fees and charges collected under securities laws."
                ),
                lookup_type="sebi_income_sources",
            ),
        )

        result = provider.lookup(query="What are the sources of income for SEBI")

        self.assertEqual(result.answer_status, "answered")
        self.assertEqual(result.sources[0].source_type, "official_web")
        self.assertTrue(result.debug["official_web_attempted"])
        self.assertFalse(result.debug["general_web_attempted"])

    def test_current_news_lookup_prefers_official_sources(self) -> None:
        provider = CurrentNewsLookupProvider(
            settings=_settings(),
            official_search_provider=_FakeOfficialWebProvider(
                answer_text="SEBI published a new official circular on 12 April 2026.",
                lookup_type="sebi_current_news",
            ),
        )

        result = provider.lookup(query="What is the latest news about SEBI")

        self.assertEqual(result.answer_status, "answered")
        self.assertTrue(all(source.source_type == "official_web" for source in result.sources))

    def test_ranking_prefers_sebi_domain_before_generic_government_matches(self) -> None:
        ranked = rank_web_sources(
            (
                WebSearchSource(
                    source_title="DEA Page",
                    source_url="https://dea.gov.in/history/sebi.html",
                    domain="dea.gov.in",
                    source_type="official_web",
                    record_key="official_web:dea.gov.in",
                ),
                WebSearchSource(
                    source_title="SEBI Archive",
                    source_url="https://www.sebi.gov.in/media-and-notifications/archives/apr-2026/test.html",
                    domain="sebi.gov.in",
                    source_type="official_web",
                    record_key="official_web:sebi.gov.in",
                ),
            ),
            allowed_domains=("sebi.gov.in", "gov.in", "nic.in"),
        )

        self.assertEqual(ranked[0].domain, "sebi.gov.in")

    def test_official_web_provider_retries_with_tool_compatible_model(self) -> None:
        attempts: list[dict[str, object]] = []

        class _FakeResponsesClient:
            def create(self, **kwargs):
                attempts.append(kwargs)
                if kwargs["model"] == "gpt-4.1-mini":
                    raise RuntimeError(
                        "Parameter 'filters' not supported with model 'gpt-4.1-mini'"
                    )
                return SimpleNamespace(
                    output=(
                        SimpleNamespace(
                            type="web_search_call",
                            action=SimpleNamespace(
                                sources=(
                                    SimpleNamespace(
                                        url="https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes"
                                    ),
                                )
                            ),
                        ),
                        SimpleNamespace(
                            type="message",
                            content=(
                                SimpleNamespace(
                                    type="output_text",
                                    text=(
                                        "ANSWER: The Chairperson of SEBI is Tuhin Kanta Pandey."
                                    ),
                                    annotations=(
                                        SimpleNamespace(
                                            type="url_citation",
                                            url="https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes",
                                            title="SEBI Official Website",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    output_text="",
                )

        class _FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.responses = _FakeResponsesClient()

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = _FakeOpenAI

        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            provider = OpenAIResponsesWebSearchProvider(
                settings=SebiOrdersRagSettings(
                    db_dsn="postgresql://unused",
                    data_root=Path(".").resolve(),
                    openai_api_key="test-key",
                    chat_model="gpt-4.1-mini",
                ),
                provider_name="official_web_search",
                default_source_type="official_web",
                default_allowed_domains=("sebi.gov.in", "gov.in"),
            )
            result = provider.search(
                request=WebSearchRequest(
                    query="Who is the chairperson of SEBI?",
                    lookup_type="sebi_current_fact",
                    instructions="Use official SEBI sources only.",
                    source_type="official_web",
                    allowed_domains=("sebi.gov.in", "gov.in"),
                )
            )

        self.assertEqual(result.answer_status, "answered")
        self.assertEqual(attempts[0]["model"], "gpt-4.1-mini")
        self.assertEqual(attempts[1]["model"], "o4-mini")
        self.assertEqual(result.debug["model_used"], "o4-mini")
        self.assertEqual(result.sources[0].domain, "sebi.gov.in")

    def test_general_web_provider_prefers_annotation_sources_and_dedupes_domains(self) -> None:
        response = SimpleNamespace(
            output=(
                SimpleNamespace(
                    type="web_search_call",
                    action=SimpleNamespace(
                        sources=(
                            SimpleNamespace(url="https://example.com/noisy-source"),
                            SimpleNamespace(url="https://www.adani.com/en/about-us/leadership/ashish-khanna"),
                        )
                    ),
                ),
                SimpleNamespace(
                    type="message",
                    content=(
                        SimpleNamespace(
                            type="output_text",
                            text="ANSWER: Ashish Khanna is the CEO of Adani Green Energy Limited.",
                            annotations=(
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://m.economictimes.com/industry/renewables/amit-singh-to-step-down-as-ceo-of-adani-green-energy-ashish-khanna-to-take-over-from-april-2025/articleshow/118760637.cms?utm_source=test",
                                    title="Amit Singh to step down as CEO of Adani Green Energy; Ashish Khanna to take over from April 2025",
                                ),
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://www.adani.com/en/about-us/leadership/ashish-khanna",
                                    title="Ashish Khanna",
                                ),
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://www.adani.com/en/our-businesses/renewable-energy",
                                    title="Adani Green Energy leadership",
                                ),
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://www.zaubacorp.com/company/ADANI-GREEN-ENERGY-LIMITED/U40106GJ2015PLC082007",
                                    title="Adani Green Energy Limited",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            output_text="",
        )

        provider = _build_fake_web_provider(response=response, provider_name="general_web_search", source_type="general_web")
        result = provider.search(
            request=WebSearchRequest(
                query="Who is the CEO of Adani Green Energy Limited?",
                lookup_type="general_knowledge",
                instructions="Answer only if the web sources support the company-role fact.",
                source_type="general_web",
            )
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertEqual(result.debug["raw_source_count"], 4)
        self.assertEqual(result.debug["filtered_source_count"], 3)
        self.assertEqual(
            {source.domain for source in result.sources},
            {"economictimes.com", "adani.com", "zaubacorp.com"},
        )
        self.assertNotIn("example.com", {source.domain for source in result.sources})
        self.assertTrue(
            all(
                (source.source_title or "").lower() != (source.domain or "").lower()
                for source in result.sources
            )
        )

    def test_general_web_provider_uses_slug_title_when_annotations_are_absent(self) -> None:
        response = SimpleNamespace(
            output=(
                SimpleNamespace(
                    type="web_search_call",
                    action=SimpleNamespace(
                        sources=(
                            SimpleNamespace(
                                url="https://www.adani.com/en/about-us/leadership/ashish-khanna/"
                            ),
                        )
                    ),
                ),
                SimpleNamespace(
                    type="message",
                    content=(
                        SimpleNamespace(
                            type="output_text",
                            text="ANSWER: Ashish Khanna is the CEO of Adani Green Energy Limited.",
                            annotations=(),
                        ),
                    ),
                ),
            ),
            output_text="",
        )

        provider = _build_fake_web_provider(response=response, provider_name="general_web_search", source_type="general_web")
        result = provider.search(
            request=WebSearchRequest(
                query="Who is the CEO of Adani Green Energy Limited?",
                lookup_type="general_knowledge",
                instructions="Answer only if the web sources support the company-role fact.",
                source_type="general_web",
            )
        )

        self.assertEqual(result.answer_status, "answered")
        self.assertEqual(result.sources[0].source_title, "Ashish Khanna")


class _EmptyStructuredFetcher:
    def fetch(self, source) -> FetchedDirectorySource:
        return FetchedDirectorySource(
            source_type=source.source_type,
            title=source.title,
            source_url=source.url,
            raw_html="<html><body></body></html>",
            content_sha256="a" * 64,
        )


class _FakeOfficialWebProvider:
    def __init__(self, *, answer_text: str, lookup_type: str) -> None:
        self._answer_text = answer_text
        self._lookup_type = lookup_type

    def search(self, *, request):
        return WebSearchResult(
            answer_status="answered",
            answer_text=self._answer_text,
            sources=(
                WebSearchSource(
                    source_title="SEBI Official Page",
                    source_url="https://www.sebi.gov.in/legal/circulars/apr-2026/test.html",
                    domain="sebi.gov.in",
                    source_type="official_web",
                    record_key="official_web:sebi.gov.in",
                ),
            ),
            provider_name="official_web_search",
            lookup_type=self._lookup_type,
            debug={"official_web_attempted": True},
        )


def _settings() -> SebiOrdersRagSettings:
    return SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
    )


def _build_fake_web_provider(*, response, provider_name: str, source_type: str) -> OpenAIResponsesWebSearchProvider:
    class _FakeResponsesClient:
        def create(self, **kwargs):
            return response

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.responses = _FakeResponsesClient()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI

    with mock.patch.dict(sys.modules, {"openai": fake_openai}):
        return OpenAIResponsesWebSearchProvider(
            settings=SebiOrdersRagSettings(
                db_dsn="postgresql://unused",
                data_root=Path(".").resolve(),
                openai_api_key="test-key",
                chat_model="o4-mini",
            ),
            provider_name=provider_name,
            default_source_type=source_type,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
