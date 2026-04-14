"""Minimal FastAPI app for local Phase 4 chat validation."""

from __future__ import annotations

from pathlib import Path

from ..config import SebiOrdersRagSettings, load_env_file
from ..logging_utils import configure_logging
from .routes_chat import router as chat_router
from ...dependencies.sebi_orders_chat_dependencies import get_sebi_orders_static_dir
from ...routers.sebi_orders_chat import router as sebi_orders_chat_router

try:  # pragma: no cover - depends on local runtime
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - depends on local runtime
    raise RuntimeError(
        "fastapi is required for the Phase 4 app. "
        "Install the dependencies from requirements-sebi-orders-rag.txt."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _mount_static_assets(app: FastAPI) -> None:
    static_dir = get_sebi_orders_static_dir()
    if any(getattr(route, "path", None) == "/static" for route in app.router.routes):
        return
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def create_app() -> FastAPI:
    """Create the standalone Phase 4 FastAPI app."""

    load_env_file(PROJECT_ROOT / ".env")
    settings = SebiOrdersRagSettings.from_env()
    configure_logging(settings.log_level)

    app = FastAPI(title="SEBI Orders RAG Phase 4")
    _mount_static_assets(app)
    app.include_router(chat_router)
    app.include_router(sebi_orders_chat_router)
    return app


app = create_app()


def main() -> None:  # pragma: no cover - runtime entrypoint
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run the Phase 4 app. "
            "Install the dependencies from requirements-sebi-orders-rag.txt."
        ) from exc

    settings = SebiOrdersRagSettings.from_env()
    uvicorn.run(
        "app.sebi_orders_rag.api.phase4_app:app",
        host=settings.phase4_app_host,
        port=settings.phase4_app_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
