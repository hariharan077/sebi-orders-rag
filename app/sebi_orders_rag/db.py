"""Database utilities for the SEBI Orders RAG retrieval store."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator

from .config import SebiOrdersRagSettings
from .exceptions import MissingDependencyError, SchemaInitializationError
from .utils.files import read_text_file

LOGGER = logging.getLogger(__name__)
_PHASE4_SCHEMA_INIT_LOCK = Lock()
_PHASE4_SCHEMA_INIT_CACHE: set[str] = set()

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection as PsycopgConnection
else:  # pragma: no cover
    PsycopgConnection = Any


def _import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise MissingDependencyError(
            "psycopg is required for SEBI Orders RAG database access. "
            "Install `psycopg[binary]` before running Phase 1."
        ) from exc
    return psycopg


@contextmanager
def get_connection(settings: SebiOrdersRagSettings) -> Iterator[PsycopgConnection]:
    """Yield a psycopg connection for the separate retrieval database."""

    psycopg = _import_psycopg()
    connection = psycopg.connect(settings.db_dsn)
    try:
        yield connection
    finally:
        connection.close()


def initialize_schema(connection: PsycopgConnection, settings: SebiOrdersRagSettings) -> None:
    """Initialize the retrieval schema from the Phase 1 SQL file."""

    if settings.embedding_dim != 3072:
        LOGGER.info(
            "Base schema file declares VECTOR(3072); configured embedding dim %s "
            "will be applied by the Phase 3 migration.",
            settings.embedding_dim,
        )

    execute_sql_file(connection, settings.sql_schema_path)


def initialize_phase4_schema(
    connection: PsycopgConnection,
    settings: SebiOrdersRagSettings,
) -> None:
    """Apply the Phase 4 session and answer metadata migration."""

    execute_sql_file(connection, settings.sql_phase4_session_metadata_path)
    initialize_directory_reference_schema(connection, settings)
    initialize_order_metadata_schema(connection, settings)
    execute_sql_file(connection, settings.sql_clarification_context_path)


def ensure_phase4_schema_initialized(settings: SebiOrdersRagSettings) -> None:
    """Initialize the Phase 4 schema once per process for one DB target."""

    cache_key = settings.db_dsn
    if cache_key in _PHASE4_SCHEMA_INIT_CACHE:
        return

    with _PHASE4_SCHEMA_INIT_LOCK:
        if cache_key in _PHASE4_SCHEMA_INIT_CACHE:
            return
        with get_connection(settings) as connection:
            initialize_phase4_schema(connection, settings)
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        _PHASE4_SCHEMA_INIT_CACHE.add(cache_key)


def initialize_directory_reference_schema(
    connection: PsycopgConnection,
    settings: SebiOrdersRagSettings,
) -> None:
    """Apply the structured SEBI directory reference migration."""

    execute_sql_file(connection, settings.sql_directory_reference_path)
    execute_sql_file(connection, settings.sql_directory_reference_hardening_path)
    execute_sql_file(connection, settings.sql_structured_info_canonical_path)


def initialize_order_metadata_schema(
    connection: PsycopgConnection,
    settings: SebiOrdersRagSettings,
) -> None:
    """Apply the extracted order-metadata tables migration."""

    execute_sql_file(connection, settings.sql_order_metadata_path)


def execute_sql_file(connection: PsycopgConnection, sql_path: Path) -> None:
    """Execute a UTF-8 SQL file against the retrieval database."""

    sql_text = read_text_file(sql_path)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_text)
    except Exception as exc:  # pragma: no cover - exercised against a real DB
        raise SchemaInitializationError(f"Failed to execute SQL from {sql_path}") from exc
