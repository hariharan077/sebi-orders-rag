"""Optional OCR helpers for low-text PDF pages."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def ocr_dependencies_available() -> bool:
    """Return True when optional OCR dependencies are importable."""

    try:
        import PIL.Image  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_page(
    page: Any,
    *,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str | None:
    """Render a PDF page and run local OCR or an OpenAI vision fallback."""

    if not ocr_dependencies_available():
        return _ocr_page_with_openai(
            page,
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )

    try:
        from PIL import Image
        import pytesseract
    except ImportError:  # pragma: no cover - guarded above
        return None

    try:
        pixmap = page.get_pixmap(alpha=False, dpi=200)
        image_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, config="--psm 6")
    except Exception as exc:  # pragma: no cover - depends on local OCR runtime
        LOGGER.warning("Local OCR failed for page %s: %s", getattr(page, "number", "?"), exc)
        return _ocr_page_with_openai(
            page,
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )

    cleaned = text.strip()
    return cleaned or None


def _ocr_page_with_openai(
    page: Any,
    *,
    model_name: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
) -> str | None:
    api_key = (openai_api_key or "").strip()
    resolved_model = (model_name or "").strip()
    if not api_key or not resolved_model:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        pixmap = page.get_pixmap(alpha=False, dpi=200)
        image_bytes = pixmap.tobytes("png")
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{image_base64}"
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 3,
            "timeout": 60.0,
        }
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=resolved_model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all visible text from this document page in reading order. "
                                "Return plain text only. Do not summarize or explain."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # pragma: no cover - depends on local OCR runtime
        LOGGER.warning(
            "OpenAI OCR fallback failed for page %s: %s",
            getattr(page, "number", "?"),
            exc,
        )
        return None

    content = response.choices[0].message.content or ""
    cleaned = content.strip()
    return cleaned or None
