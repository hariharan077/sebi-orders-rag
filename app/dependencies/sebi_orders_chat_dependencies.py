"""Dependency helpers for the thin SEBI Orders portal chat surface."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..sebi_orders_rag.config import SebiOrdersRagSettings, load_env_file
from ..services.sebi_orders_rag_service import SebiOrdersRagService

try:  # pragma: no cover - runtime import
    from fastapi.templating import Jinja2Templates
except ImportError as exc:  # pragma: no cover - depends on runtime
    raise RuntimeError(
        "fastapi is required for the SEBI Orders portal chat integration. "
        "Install the dependencies from requirements-sebi-orders-rag.txt."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = PROJECT_ROOT / ".env"
_TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
_STATIC_DIR = PROJECT_ROOT / "app" / "static"


def load_sebi_orders_chat_settings() -> SebiOrdersRagSettings:
    """Load runtime settings for the portal chat wrapper."""

    load_env_file(_ENV_FILE)
    return SebiOrdersRagSettings.from_env()


def get_sebi_orders_chat_service() -> SebiOrdersRagService:
    """Construct the thin portal adapter over the existing RAG pipeline."""

    return SebiOrdersRagService(settings_loader=load_sebi_orders_chat_settings)


@lru_cache(maxsize=1)
def get_sebi_orders_templates() -> Jinja2Templates:
    """Return cached Jinja templates for the portal chat page."""

    return Jinja2Templates(directory=str(_TEMPLATES_DIR))


def get_sebi_orders_static_dir() -> Path:
    """Return the static asset root used by the portal chat page."""

    return _STATIC_DIR
