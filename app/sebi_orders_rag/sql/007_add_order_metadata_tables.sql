CREATE TABLE IF NOT EXISTS order_metadata (
    order_metadata_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL UNIQUE REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    signatory_name TEXT,
    signatory_designation TEXT,
    signatory_page_start INT,
    signatory_page_end INT,
    order_date DATE,
    place TEXT,
    issuing_authority_type TEXT,
    authority_panel TEXT[] NOT NULL DEFAULT '{}',
    metadata_confidence NUMERIC(5,4),
    extraction_version TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_legal_provisions (
    provision_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    statute_name TEXT,
    section_or_regulation TEXT,
    provision_type TEXT,
    text_snippet TEXT,
    page_start INT,
    page_end INT,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_numeric_facts (
    numeric_fact_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    fact_type TEXT NOT NULL,
    subject TEXT,
    value_text TEXT,
    value_numeric DOUBLE PRECISION,
    unit TEXT,
    context_label TEXT,
    page_start INT,
    page_end INT,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_price_movements (
    price_movement_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    period_label TEXT NOT NULL,
    period_start_text TEXT,
    period_end_text TEXT,
    start_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    end_price DOUBLE PRECISION,
    pct_change DOUBLE PRECISION,
    rationale TEXT,
    page_start INT,
    page_end INT,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_legal_provisions_doc_sha
    ON order_legal_provisions(document_version_id, row_sha256);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_numeric_facts_doc_sha
    ON order_numeric_facts(document_version_id, row_sha256);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_price_movements_doc_sha
    ON order_price_movements(document_version_id, row_sha256);

CREATE INDEX IF NOT EXISTS idx_order_metadata_signatory_name
    ON order_metadata USING GIN (signatory_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_order_metadata_authority_type
    ON order_metadata(issuing_authority_type);

CREATE INDEX IF NOT EXISTS idx_order_legal_provisions_doc_id
    ON order_legal_provisions(document_version_id);

CREATE INDEX IF NOT EXISTS idx_order_legal_provisions_statute
    ON order_legal_provisions USING GIN (statute_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_order_legal_provisions_reference
    ON order_legal_provisions USING GIN (section_or_regulation gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_order_legal_provisions_pages
    ON order_legal_provisions(page_start, page_end);

CREATE INDEX IF NOT EXISTS idx_order_numeric_facts_doc_id
    ON order_numeric_facts(document_version_id);

CREATE INDEX IF NOT EXISTS idx_order_numeric_facts_fact_type
    ON order_numeric_facts(fact_type);

CREATE INDEX IF NOT EXISTS idx_order_numeric_facts_context
    ON order_numeric_facts USING GIN (context_label gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_order_price_movements_doc_id
    ON order_price_movements(document_version_id);
