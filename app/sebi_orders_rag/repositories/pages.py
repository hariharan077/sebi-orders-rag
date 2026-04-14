"""Page persistence helpers for extracted SEBI order text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..schemas import ExtractedPage


class DocumentPageRepository:
    """Replace page rows for a single document version atomically."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def replace_pages(
        self,
        *,
        document_version_id: int,
        pages: Sequence[ExtractedPage],
    ) -> int:
        """Delete and reinsert all page rows for a document version."""

        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_pages WHERE document_version_id = %s",
                (document_version_id,),
            )
            if not pages:
                return 0

            cursor.executemany(
                """
                INSERT INTO document_pages (
                    document_version_id,
                    page_no,
                    extracted_text,
                    ocr_text,
                    final_text,
                    char_count,
                    token_count,
                    low_text,
                    page_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        document_version_id,
                        page.page_no,
                        page.extracted_text,
                        page.ocr_text,
                        page.final_text,
                        page.char_count,
                        page.token_count,
                        page.low_text,
                        page.page_sha256,
                    )
                    for page in pages
                ],
            )
        return len(pages)
