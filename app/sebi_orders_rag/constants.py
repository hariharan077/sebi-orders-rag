"""Constants shared across the SEBI Orders RAG module."""

from __future__ import annotations

ENV_PREFIX = "SEBI_ORDERS_RAG_"
MANIFEST_FILE_NAME = "orders_manifest.csv"
MANIFEST_COLUMNS = (
    "record_key",
    "bucket_name",
    "order_date",
    "title",
    "external_record_id",
    "detail_url",
    "pdf_url",
    "local_filename",
    "status",
    "error",
    "first_seen_at",
    "last_seen_at",
)
REQUIRED_MANIFEST_COLUMNS = (
    "record_key",
    "bucket_name",
    "title",
    "status",
    "first_seen_at",
    "last_seen_at",
)
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_PARSER_NAME = "sebi_orders_pdf_parser"
DEFAULT_PARSER_VERSION = "v1"
DEFAULT_CHUNKING_VERSION = "v2.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
SQL_INIT_SCHEMA_FILE = "001_init_schema.sql"
SQL_CHUNKING_METADATA_FILE = "002_add_chunking_metadata.sql"
SQL_HIERARCHICAL_RETRIEVAL_FILE = "003_add_hierarchical_retrieval.sql"
HASH_CHUNK_SIZE_BYTES = 1024 * 1024
PENDING_STATUS = "pending"
FAILED_STATUS = "failed"
PROCESSING_STATUS = "processing"
DONE_STATUS = "done"
DEFAULT_ENABLE_OCR = False
DEFAULT_LOW_TEXT_CHAR_THRESHOLD = 80
DEFAULT_TARGET_CHUNK_TOKENS = 800
DEFAULT_MAX_CHUNK_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
DEFAULT_MIN_HEADING_CAPS_RATIO = 0.60
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_RETRIEVAL_TOP_K_DOCS = 10
DEFAULT_RETRIEVAL_TOP_K_SECTIONS = 20
DEFAULT_RETRIEVAL_TOP_K_CHUNKS = 25
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_CHAT_TEMPERATURE = 0.0
DEFAULT_MAX_CONTEXT_CHUNKS = 8
DEFAULT_MAX_CONTEXT_TOKENS = 12000
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_ENABLE_MEMORY = True
DEFAULT_CURRENT_LOOKUP_ENABLED = True
DEFAULT_CURRENT_LOOKUP_TIMEOUT_SECONDS = 10.0
DEFAULT_WEB_FALLBACK_ENABLED = True
DEFAULT_OFFICIAL_WEB_SEARCH_ENABLED = True
DEFAULT_GENERAL_WEB_SEARCH_ENABLED = True
DEFAULT_WEB_SEARCH_PROVIDER = "openai"
DEFAULT_OFFICIAL_ALLOWED_DOMAINS = (
    "sebi.gov.in",
    "gov.in",
    "nic.in",
)
DEFAULT_GENERAL_ALLOWED_DOMAINS: tuple[str, ...] = ()
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 20.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 6
DEFAULT_DIRECTORY_ENABLED = True
DEFAULT_DIRECTORY_REFRESH_ENABLED = True
DEFAULT_DIRECTORY_TIMEOUT_SECONDS = 20.0
DEFAULT_DIRECTORY_USER_AGENT = "SEBIOrdersRAG/1.0 (+internal-portal)"
DEFAULT_DIRECTORY_SOURCE_DIRECTORY_URL = "https://www.sebi.gov.in/sebiweb/home/directory-of-sebi.jsp"
DEFAULT_DIRECTORY_SOURCE_ORGCHART_URL = "https://www.sebi.gov.in/orgchart-grid.html"
DEFAULT_DIRECTORY_SOURCE_REGIONAL_OFFICES_URL = (
    "https://www.sebi.gov.in/department/regional-offices-43/contact.html"
)
DEFAULT_DIRECTORY_SOURCE_CONTACT_US_URL = "https://www.sebi.gov.in/contact-us.html"
DEFAULT_DIRECTORY_SOURCE_BOARD_MEMBERS_URL = (
    "https://www.sebi.gov.in/sebiweb/boardmember/BoardMemberAction.do?doBoardMember=yes&lang=en"
)
DEFAULT_PHASE4_APP_HOST = "127.0.0.1"
DEFAULT_PHASE4_APP_PORT = 8014
DEFAULT_CONTROL_PACK_DIR_PREFIX = "sebi_control_pack_"
SQL_PHASE4_SESSION_METADATA_FILE = "004_add_phase4_session_metadata.sql"
SQL_DIRECTORY_REFERENCE_FILE = "005_add_directory_reference_tables.sql"
SQL_DIRECTORY_REFERENCE_HARDENING_FILE = "006_add_board_members_and_reference_views.sql"
SQL_ORDER_METADATA_FILE = "007_add_order_metadata_tables.sql"
SQL_STRUCTURED_INFO_CANONICAL_FILE = "008_add_canonical_reference_views_and_counts.sql"
SQL_CLARIFICATION_CONTEXT_FILE = "009_add_clarification_context.sql"
PHASE4_ROUTE_MODES = (
    "smalltalk",
    "general_knowledge",
    "structured_current_info",
    "current_official_lookup",
    "current_news_lookup",
    "historical_official_lookup",
    "direct_llm",
    "exact_lookup",
    "hierarchical_rag",
    "memory_scoped_rag",
    "corpus_metadata",
    "clarify",
    "abstain",
)
PLANNER_ROUTES = (
    "structured_current_info",
    "order_metadata",
    "order_corpus_rag",
    "official_web",
    "current_news",
    "general_knowledge",
    "clarify",
    "abstain",
)
LEGACY_RETRIEVAL_LOG_MODE_BY_ROUTE = {
    "smalltalk": "direct_llm",
    "general_knowledge": "direct_llm",
    "structured_current_info": "direct_llm",
    "current_official_lookup": "direct_llm",
    "current_news_lookup": "direct_llm",
    "historical_official_lookup": "direct_llm",
    "direct_llm": "direct_llm",
    "exact_lookup": "db_search",
    "hierarchical_rag": "db_search",
    "memory_scoped_rag": "memory_scoped_db_search",
    "corpus_metadata": "direct_llm",
    "clarify": "direct_llm",
    "abstain": "abstain",
}
SUBSTANTIVE_SECTION_TYPES = (
    "operative_order",
    "directions",
    "findings",
)
