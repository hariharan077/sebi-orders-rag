"""Runtime configuration for the SEBI Orders RAG module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHUNKING_VERSION,
    DEFAULT_CONTROL_PACK_DIR_PREFIX,
    DEFAULT_DIRECTORY_ENABLED,
    DEFAULT_DIRECTORY_REFRESH_ENABLED,
    DEFAULT_DIRECTORY_SOURCE_BOARD_MEMBERS_URL,
    DEFAULT_DIRECTORY_SOURCE_CONTACT_US_URL,
    DEFAULT_DIRECTORY_SOURCE_DIRECTORY_URL,
    DEFAULT_DIRECTORY_SOURCE_ORGCHART_URL,
    DEFAULT_DIRECTORY_SOURCE_REGIONAL_OFFICES_URL,
    DEFAULT_DIRECTORY_TIMEOUT_SECONDS,
    DEFAULT_DIRECTORY_USER_AGENT,
    DEFAULT_CURRENT_LOOKUP_ENABLED,
    DEFAULT_CURRENT_LOOKUP_TIMEOUT_SECONDS,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENABLE_MEMORY,
    DEFAULT_ENABLE_OCR,
    DEFAULT_GENERAL_WEB_SEARCH_ENABLED,
    DEFAULT_GENERAL_ALLOWED_DOMAINS,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOW_TEXT_CHAR_THRESHOLD,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_MIN_HEADING_CAPS_RATIO,
    DEFAULT_OFFICIAL_ALLOWED_DOMAINS,
    DEFAULT_OFFICIAL_WEB_SEARCH_ENABLED,
    DEFAULT_PARSER_NAME,
    DEFAULT_PARSER_VERSION,
    DEFAULT_PHASE4_APP_HOST,
    DEFAULT_PHASE4_APP_PORT,
    DEFAULT_RETRIEVAL_TOP_K_CHUNKS,
    DEFAULT_RETRIEVAL_TOP_K_DOCS,
    DEFAULT_RETRIEVAL_TOP_K_SECTIONS,
    DEFAULT_TARGET_CHUNK_TOKENS,
    DEFAULT_WEB_FALLBACK_ENABLED,
    DEFAULT_WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_PROVIDER,
    DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
    ENV_PREFIX,
    SQL_CHUNKING_METADATA_FILE,
    SQL_DIRECTORY_REFERENCE_HARDENING_FILE,
    SQL_DIRECTORY_REFERENCE_FILE,
    SQL_HIERARCHICAL_RETRIEVAL_FILE,
    SQL_INIT_SCHEMA_FILE,
    SQL_CLARIFICATION_CONTEXT_FILE,
    SQL_ORDER_METADATA_FILE,
    SQL_PHASE4_SESSION_METADATA_FILE,
    SQL_STRUCTURED_INFO_CANONICAL_FILE,
)
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class SebiOrdersRagSettings:
    """Settings for the SEBI Orders RAG retrieval module."""

    db_dsn: str
    data_root: Path
    log_level: str = DEFAULT_LOG_LEVEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    parser_version: str = DEFAULT_PARSER_VERSION
    chunking_version: str = DEFAULT_CHUNKING_VERSION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    enable_ocr: bool = DEFAULT_ENABLE_OCR
    low_text_char_threshold: int = DEFAULT_LOW_TEXT_CHAR_THRESHOLD
    target_chunk_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    min_heading_caps_ratio: float = DEFAULT_MIN_HEADING_CAPS_RATIO
    retrieval_top_k_docs: int = DEFAULT_RETRIEVAL_TOP_K_DOCS
    retrieval_top_k_sections: int = DEFAULT_RETRIEVAL_TOP_K_SECTIONS
    retrieval_top_k_chunks: int = DEFAULT_RETRIEVAL_TOP_K_CHUNKS
    chat_model: str = DEFAULT_CHAT_MODEL
    chat_temperature: float = DEFAULT_CHAT_TEMPERATURE
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
    enable_memory: bool = DEFAULT_ENABLE_MEMORY
    current_lookup_enabled: bool = DEFAULT_CURRENT_LOOKUP_ENABLED
    current_lookup_timeout_seconds: float = DEFAULT_CURRENT_LOOKUP_TIMEOUT_SECONDS
    web_fallback_enabled: bool = DEFAULT_WEB_FALLBACK_ENABLED
    official_web_search_enabled: bool = DEFAULT_OFFICIAL_WEB_SEARCH_ENABLED
    general_web_search_enabled: bool = DEFAULT_GENERAL_WEB_SEARCH_ENABLED
    web_search_provider: str = DEFAULT_WEB_SEARCH_PROVIDER
    official_allowed_domains: tuple[str, ...] = DEFAULT_OFFICIAL_ALLOWED_DOMAINS
    general_allowed_domains: tuple[str, ...] = DEFAULT_GENERAL_ALLOWED_DOMAINS
    web_search_timeout_seconds: float = DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS
    web_search_max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS
    directory_enabled: bool = DEFAULT_DIRECTORY_ENABLED
    directory_refresh_enabled: bool = DEFAULT_DIRECTORY_REFRESH_ENABLED
    directory_timeout_seconds: float = DEFAULT_DIRECTORY_TIMEOUT_SECONDS
    directory_user_agent: str = DEFAULT_DIRECTORY_USER_AGENT
    directory_source_directory_url: str = DEFAULT_DIRECTORY_SOURCE_DIRECTORY_URL
    directory_source_orgchart_url: str = DEFAULT_DIRECTORY_SOURCE_ORGCHART_URL
    directory_source_regional_offices_url: str = DEFAULT_DIRECTORY_SOURCE_REGIONAL_OFFICES_URL
    directory_source_contact_us_url: str = DEFAULT_DIRECTORY_SOURCE_CONTACT_US_URL
    directory_source_board_members_url: str = DEFAULT_DIRECTORY_SOURCE_BOARD_MEMBERS_URL
    phase4_app_host: str = DEFAULT_PHASE4_APP_HOST
    phase4_app_port: int = DEFAULT_PHASE4_APP_PORT
    control_pack_root: Path | None = None

    @property
    def parser_name(self) -> str:
        """Stable parser identifier recorded for planned versions."""

        return DEFAULT_PARSER_NAME

    @property
    def sql_schema_path(self) -> Path:
        """Absolute path to the Phase 1 schema file."""

        return Path(__file__).resolve().parent / "sql" / SQL_INIT_SCHEMA_FILE

    @property
    def sql_chunking_metadata_path(self) -> Path:
        """Absolute path to the Phase 2 chunking metadata migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_CHUNKING_METADATA_FILE

    @property
    def sql_hierarchical_retrieval_path(self) -> Path:
        """Absolute path to the Phase 3 hierarchical retrieval migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_HIERARCHICAL_RETRIEVAL_FILE

    @property
    def sql_phase4_session_metadata_path(self) -> Path:
        """Absolute path to the Phase 4 session metadata migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_PHASE4_SESSION_METADATA_FILE

    @property
    def sql_directory_reference_path(self) -> Path:
        """Absolute path to the structured directory reference migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_DIRECTORY_REFERENCE_FILE

    @property
    def sql_directory_reference_hardening_path(self) -> Path:
        """Absolute path to the structured-reference hardening migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_DIRECTORY_REFERENCE_HARDENING_FILE

    @property
    def sql_order_metadata_path(self) -> Path:
        """Absolute path to the order-metadata extraction migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_ORDER_METADATA_FILE

    @property
    def sql_structured_info_canonical_path(self) -> Path:
        """Absolute path to the canonical structured-info migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_STRUCTURED_INFO_CANONICAL_FILE

    @property
    def sql_clarification_context_path(self) -> Path:
        """Absolute path to the clarification-context migration."""

        return Path(__file__).resolve().parent / "sql" / SQL_CLARIFICATION_CONTEXT_FILE

    @classmethod
    def from_env(
        cls,
        *,
        data_root_override: str | os.PathLike[str] | None = None,
        control_pack_root_override: str | os.PathLike[str] | None = None,
    ) -> "SebiOrdersRagSettings":
        """Build settings from environment variables and optional CLI overrides."""

        db_dsn = _read_required_env("DB_DSN")
        data_root_value = (
            str(data_root_override)
            if data_root_override is not None
            else _read_required_env("DATA_ROOT")
        )

        embedding_dim_value = os.environ.get(
            f"{ENV_PREFIX}EMBEDDING_DIM",
            str(DEFAULT_EMBEDDING_DIM),
        )
        try:
            embedding_dim = int(embedding_dim_value)
        except ValueError as exc:
            raise ConfigurationError(
                f"{ENV_PREFIX}EMBEDDING_DIM must be an integer"
            ) from exc
        if embedding_dim <= 0:
            raise ConfigurationError(
                f"{ENV_PREFIX}EMBEDDING_DIM must be greater than zero"
            )

        low_text_char_threshold = _read_int_env(
            "LOW_TEXT_CHAR_THRESHOLD",
            DEFAULT_LOW_TEXT_CHAR_THRESHOLD,
            minimum=0,
        )
        target_chunk_tokens = _read_int_env(
            "TARGET_CHUNK_TOKENS",
            DEFAULT_TARGET_CHUNK_TOKENS,
            minimum=1,
        )
        max_chunk_tokens = _read_int_env(
            "MAX_CHUNK_TOKENS",
            DEFAULT_MAX_CHUNK_TOKENS,
            minimum=1,
        )
        chunk_overlap_tokens = _read_int_env(
            "CHUNK_OVERLAP_TOKENS",
            DEFAULT_CHUNK_OVERLAP_TOKENS,
            minimum=0,
        )
        min_heading_caps_ratio = _read_float_env(
            "MIN_HEADING_CAPS_RATIO",
            DEFAULT_MIN_HEADING_CAPS_RATIO,
            minimum=0.0,
            maximum=1.0,
        )

        if target_chunk_tokens > max_chunk_tokens:
            raise ConfigurationError(
                f"{ENV_PREFIX}TARGET_CHUNK_TOKENS must be less than or equal to "
                f"{ENV_PREFIX}MAX_CHUNK_TOKENS"
            )
        if chunk_overlap_tokens >= max_chunk_tokens:
            raise ConfigurationError(
                f"{ENV_PREFIX}CHUNK_OVERLAP_TOKENS must be smaller than "
                f"{ENV_PREFIX}MAX_CHUNK_TOKENS"
            )

        embed_batch_size = _read_int_env(
            "EMBED_BATCH_SIZE",
            DEFAULT_EMBED_BATCH_SIZE,
            minimum=1,
        )
        retrieval_top_k_docs = _read_int_env(
            "RETRIEVAL_TOP_K_DOCS",
            DEFAULT_RETRIEVAL_TOP_K_DOCS,
            minimum=1,
        )
        retrieval_top_k_sections = _read_int_env(
            "RETRIEVAL_TOP_K_SECTIONS",
            DEFAULT_RETRIEVAL_TOP_K_SECTIONS,
            minimum=1,
        )
        retrieval_top_k_chunks = _read_int_env(
            "RETRIEVAL_TOP_K_CHUNKS",
            DEFAULT_RETRIEVAL_TOP_K_CHUNKS,
            minimum=1,
        )
        max_context_chunks = _read_int_env(
            "MAX_CONTEXT_CHUNKS",
            DEFAULT_MAX_CONTEXT_CHUNKS,
            minimum=1,
        )
        max_context_tokens = _read_int_env(
            "MAX_CONTEXT_TOKENS",
            DEFAULT_MAX_CONTEXT_TOKENS,
            minimum=1,
        )
        low_confidence_threshold = _read_float_env(
            "LOW_CONFIDENCE_THRESHOLD",
            DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        )
        phase4_app_port = _read_int_env(
            "PHASE4_APP_PORT",
            DEFAULT_PHASE4_APP_PORT,
            minimum=1,
        )
        control_pack_root = _resolve_control_pack_root(control_pack_root_override)

        return cls(
            db_dsn=db_dsn,
            data_root=Path(data_root_value).expanduser().resolve(strict=False),
            log_level=os.environ.get(
                f"{ENV_PREFIX}LOG_LEVEL",
                DEFAULT_LOG_LEVEL,
            ).upper(),
            embedding_dim=embedding_dim,
            parser_version=os.environ.get(
                f"{ENV_PREFIX}PARSER_VERSION",
                DEFAULT_PARSER_VERSION,
            ),
            chunking_version=os.environ.get(
                f"{ENV_PREFIX}CHUNKING_VERSION",
                DEFAULT_CHUNKING_VERSION,
            ),
            embedding_model=os.environ.get(
                f"{ENV_PREFIX}EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            ),
            openai_api_key=_read_optional_env("OPENAI_API_KEY"),
            openai_base_url=_read_optional_env("OPENAI_BASE_URL"),
            embed_batch_size=embed_batch_size,
            enable_ocr=_read_bool_env("ENABLE_OCR", DEFAULT_ENABLE_OCR),
            low_text_char_threshold=low_text_char_threshold,
            target_chunk_tokens=target_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            min_heading_caps_ratio=min_heading_caps_ratio,
            retrieval_top_k_docs=retrieval_top_k_docs,
            retrieval_top_k_sections=retrieval_top_k_sections,
            retrieval_top_k_chunks=retrieval_top_k_chunks,
            chat_model=os.environ.get(
                f"{ENV_PREFIX}CHAT_MODEL",
                DEFAULT_CHAT_MODEL,
            ).strip()
            or DEFAULT_CHAT_MODEL,
            chat_temperature=_read_float_env(
                "CHAT_TEMPERATURE",
                DEFAULT_CHAT_TEMPERATURE,
                minimum=0.0,
                maximum=2.0,
            ),
            max_context_chunks=max_context_chunks,
            max_context_tokens=max_context_tokens,
            low_confidence_threshold=low_confidence_threshold,
            enable_memory=_read_bool_env("ENABLE_MEMORY", DEFAULT_ENABLE_MEMORY),
            current_lookup_enabled=_read_bool_env(
                "CURRENT_LOOKUP_ENABLED",
                DEFAULT_CURRENT_LOOKUP_ENABLED,
            ),
            current_lookup_timeout_seconds=_read_float_env(
                "CURRENT_LOOKUP_TIMEOUT_SECONDS",
                DEFAULT_CURRENT_LOOKUP_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=60.0,
            ),
            web_fallback_enabled=_read_bool_env(
                "WEB_FALLBACK_ENABLED",
                DEFAULT_WEB_FALLBACK_ENABLED,
            ),
            official_web_search_enabled=_read_bool_env(
                "OFFICIAL_WEB_SEARCH_ENABLED",
                DEFAULT_OFFICIAL_WEB_SEARCH_ENABLED,
            ),
            general_web_search_enabled=_read_bool_env(
                "GENERAL_WEB_SEARCH_ENABLED",
                DEFAULT_GENERAL_WEB_SEARCH_ENABLED,
            ),
            web_search_provider=(
                os.environ.get(
                    f"{ENV_PREFIX}WEB_SEARCH_PROVIDER",
                    DEFAULT_WEB_SEARCH_PROVIDER,
                ).strip().lower()
                or DEFAULT_WEB_SEARCH_PROVIDER
            ),
            official_allowed_domains=_read_csv_env(
                "OFFICIAL_ALLOWED_DOMAINS",
                DEFAULT_OFFICIAL_ALLOWED_DOMAINS,
            ),
            general_allowed_domains=_read_csv_env(
                "GENERAL_WEB_ALLOWED_DOMAINS",
                DEFAULT_GENERAL_ALLOWED_DOMAINS,
            ),
            web_search_timeout_seconds=_read_float_env(
                "WEB_SEARCH_TIMEOUT_SECONDS",
                DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=120.0,
            ),
            web_search_max_results=_read_int_env(
                "WEB_SEARCH_MAX_RESULTS",
                DEFAULT_WEB_SEARCH_MAX_RESULTS,
                minimum=1,
            ),
            directory_enabled=_read_bool_env(
                "DIRECTORY_ENABLED",
                DEFAULT_DIRECTORY_ENABLED,
            ),
            directory_refresh_enabled=_read_bool_env(
                "DIRECTORY_REFRESH_ENABLED",
                DEFAULT_DIRECTORY_REFRESH_ENABLED,
            ),
            directory_timeout_seconds=_read_float_env(
                "DIRECTORY_TIMEOUT_SECONDS",
                DEFAULT_DIRECTORY_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=60.0,
            ),
            directory_user_agent=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_USER_AGENT",
                DEFAULT_DIRECTORY_USER_AGENT,
            ).strip()
            or DEFAULT_DIRECTORY_USER_AGENT,
            directory_source_directory_url=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_SOURCE_DIRECTORY_URL",
                DEFAULT_DIRECTORY_SOURCE_DIRECTORY_URL,
            ).strip()
            or DEFAULT_DIRECTORY_SOURCE_DIRECTORY_URL,
            directory_source_orgchart_url=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_SOURCE_ORGCHART_URL",
                DEFAULT_DIRECTORY_SOURCE_ORGCHART_URL,
            ).strip()
            or DEFAULT_DIRECTORY_SOURCE_ORGCHART_URL,
            directory_source_regional_offices_url=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_SOURCE_REGIONAL_OFFICES_URL",
                DEFAULT_DIRECTORY_SOURCE_REGIONAL_OFFICES_URL,
            ).strip()
            or DEFAULT_DIRECTORY_SOURCE_REGIONAL_OFFICES_URL,
            directory_source_contact_us_url=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_SOURCE_CONTACT_US_URL",
                DEFAULT_DIRECTORY_SOURCE_CONTACT_US_URL,
            ).strip()
            or DEFAULT_DIRECTORY_SOURCE_CONTACT_US_URL,
            directory_source_board_members_url=os.environ.get(
                f"{ENV_PREFIX}DIRECTORY_SOURCE_BOARD_MEMBERS_URL",
                DEFAULT_DIRECTORY_SOURCE_BOARD_MEMBERS_URL,
            ).strip()
            or DEFAULT_DIRECTORY_SOURCE_BOARD_MEMBERS_URL,
            phase4_app_host=os.environ.get(
                f"{ENV_PREFIX}PHASE4_APP_HOST",
                DEFAULT_PHASE4_APP_HOST,
            ).strip()
            or DEFAULT_PHASE4_APP_HOST,
            phase4_app_port=phase4_app_port,
            control_pack_root=control_pack_root,
        )


def _read_required_env(name: str) -> str:
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        raise ConfigurationError(f"Missing required environment variable: {env_name}")
    return value.strip()


def _read_optional_env(name: str) -> str | None:
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_control_pack_root(
    override: str | os.PathLike[str] | None,
) -> Path | None:
    if override is not None:
        return _validate_optional_path(Path(override))

    configured = _read_optional_env("CONTROL_PACK_ROOT")
    if configured:
        return _validate_optional_path(Path(configured))

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"
    if not artifacts_dir.exists():
        return None

    candidates = [
        path
        for path in artifacts_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(DEFAULT_CONTROL_PACK_DIR_PREFIX)
        and "smoke" not in path.name
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _validate_optional_path(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.exists():
        raise ConfigurationError(f"Configured path does not exist: {resolved}")
    return resolved


def load_env_file(path: Path, *, overwrite: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a local `.env` file."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if not overwrite and key in os.environ:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _read_bool_env(name: str, default: bool) -> bool:
    env_name = f"{ENV_PREFIX}{name}"
    value = os.environ.get(env_name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{env_name} must be a boolean")


def _read_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    env_name = f"{ENV_PREFIX}{name}"
    raw_value = os.environ.get(env_name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be an integer") from exc

    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{env_name} must be greater than or equal to {minimum}")
    return value


def _read_float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    env_name = f"{ENV_PREFIX}{name}"
    raw_value = os.environ.get(env_name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be a number") from exc

    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{env_name} must be greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{env_name} must be less than or equal to {maximum}")
    return value


def _read_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    env_name = f"{ENV_PREFIX}{name}"
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default
    values = tuple(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )
    return values or default
