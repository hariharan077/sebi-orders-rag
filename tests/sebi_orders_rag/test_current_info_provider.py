from __future__ import annotations

import unittest
from pathlib import Path

try:
    import bs4  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    bs4 = None

from app.sebi_orders_rag.config import SebiOrdersRagSettings
from app.sebi_orders_rag.current_info.official_lookup import OfficialWebsiteCurrentInfoProvider
from app.sebi_orders_rag.directory_data.models import FetchedDirectorySource


@unittest.skipIf(bs4 is None, "bs4 is not installed in this environment")
class CurrentInfoProviderTests(unittest.TestCase):
    def test_answers_chairperson_from_live_structured_pages(self) -> None:
        provider = OfficialWebsiteCurrentInfoProvider(
            settings=_settings(),
            fetcher=_FakeFetcher(
                {
                    "https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp": """
                        <html><body>
                            <div class="portlet1 box1 green">
                                <div class="portlet-title"><h2>HEAD OFFICE, MUMBAI</h2></div>
                                <table class="table1">
                                    <thead>
                                        <tr><th colspan="5"><h3>Chairman</h3></th></tr>
                                        <tr>
                                            <th>Staff No</th><th>Name</th><th>Date of Joining</th><th>Email ID</th><th>Telephone No</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td>2801</td><td>TUHIN KANTA PANDEY</td><td>Mar 01, 2025</td><td>chairman [at] sebi [dot] gov [dot] in</td><td>022-40459999</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </body></html>
                    """,
                    "https://www.sebi.gov.in/orgchart-grid.html": """
                        <html><body>
                            <div class="orgchart">
                                <div class="tree-m-info">
                                    <h3>Shri. Tuhin Kanta Pandey</h3>
                                    <h4>Chairman</h4>
                                    <h5>chairman@sebi.gov.in</h5>
                                </div>
                            </div>
                        </body></html>
                    """,
                    "https://www.sebi.gov.in/department/regional-offices-43/contact.html": "<html><body></body></html>",
                    "https://www.sebi.gov.in/contact-us.html": "<html><body><script>var locations = [];</script></body></html>",
                }
            ),
        )

        result = provider.lookup(query="Who is the chairperson of SEBI?")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Tuhin Kanta Pandey", result.answer_text)
        self.assertEqual(result.sources[0].record_key, "official:directory")

    def test_answers_ministry_relationship_from_official_page(self) -> None:
        provider = OfficialWebsiteCurrentInfoProvider(
            settings=_settings(),
            fetcher=_FakeFetcher(
                {
                    "https://dea.gov.in/index.php/our-organisations/department-economic-affairs": """
                        <html><body>
                            <div>Government of India Ministry of Finance Department of Economic Affairs</div>
                            <h2>Institutions under Department of Economic Affairs</h2>
                            <h3>Securities and Exchange Board of India (SEBI)</h3>
                        </body></html>
                    """,
                    "https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp": "<html></html>",
                    "https://www.sebi.gov.in/orgchart-grid.html": "<html></html>",
                    "https://www.sebi.gov.in/department/regional-offices-43/contact.html": "<html></html>",
                    "https://www.sebi.gov.in/contact-us.html": "<html><script>var locations = [];</script></html>",
                }
            ),
        )

        result = provider.lookup(query="Does SEBI come under Ministry of Finance?")

        self.assertEqual(result.answer_status, "answered")
        self.assertIn("Department of Economic Affairs", result.answer_text)
        self.assertEqual(result.sources[0].record_key, "official:dea.gov.in")

    def test_returns_unavailable_when_ministry_lookup_fails(self) -> None:
        provider = OfficialWebsiteCurrentInfoProvider(
            settings=_settings(),
            fetcher=_FakeFetcher(
                {
                    "https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp": "<html></html>",
                    "https://www.sebi.gov.in/orgchart-grid.html": "<html></html>",
                    "https://www.sebi.gov.in/department/regional-offices-43/contact.html": "<html></html>",
                    "https://www.sebi.gov.in/contact-us.html": "<html><script>var locations = [];</script></html>",
                },
                failing_urls={"https://dea.gov.in/index.php/our-organisations/department-economic-affairs"},
            ),
        )

        result = provider.lookup(query="Which ministry does SEBI come under?")

        self.assertEqual(result.answer_status, "unavailable")
        self.assertIn("unavailable", result.answer_text.lower())


class _FakeFetcher:
    def __init__(self, payload_by_url: dict[str, str], failing_urls: set[str] | None = None) -> None:
        self._payload_by_url = payload_by_url
        self._failing_urls = failing_urls or set()

    def fetch(self, source) -> FetchedDirectorySource:
        if source.url in self._failing_urls:
            raise RuntimeError("network unavailable")
        payload = self._payload_by_url[source.url]
        return FetchedDirectorySource(
            source_type=source.source_type,
            title=source.title,
            source_url=source.url,
            raw_html=payload,
            content_sha256="a" * 64,
        )


def _settings() -> SebiOrdersRagSettings:
    return SebiOrdersRagSettings(
        db_dsn="postgresql://unused",
        data_root=Path(".").resolve(),
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
