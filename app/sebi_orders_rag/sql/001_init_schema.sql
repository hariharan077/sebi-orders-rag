CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS source_documents (
    document_id BIGSERIAL PRIMARY KEY,
    record_key TEXT NOT NULL UNIQUE,
    bucket_name TEXT NOT NULL,
    external_record_id TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    current_version_id BIGINT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS document_versions (
    document_version_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES source_documents(document_id),
    order_date DATE,
    title TEXT NOT NULL,
    detail_url TEXT,
    pdf_url TEXT NOT NULL,
    local_filename TEXT NOT NULL,
    local_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_sha256 CHAR(64) NOT NULL,
    manifest_status TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    extraction_status TEXT NOT NULL CHECK (extraction_status IN ('pending','processing','done','failed')),
    ocr_used BOOLEAN NOT NULL DEFAULT FALSE,
    page_count INT,
    extracted_char_count INT,
    ingest_status TEXT NOT NULL CHECK (ingest_status IN ('pending','processing','done','failed')),
    ingest_error TEXT,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, file_sha256)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_source_documents_current_version'
    ) THEN
        ALTER TABLE source_documents
            ADD CONSTRAINT fk_source_documents_current_version
            FOREIGN KEY (current_version_id)
            REFERENCES document_versions(document_version_id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS document_pages (
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    page_no INT NOT NULL,
    extracted_text TEXT,
    ocr_text TEXT,
    final_text TEXT,
    char_count INT NOT NULL DEFAULT 0,
    token_count INT NOT NULL DEFAULT 0,
    low_text BOOLEAN NOT NULL DEFAULT FALSE,
    page_sha256 CHAR(64),
    PRIMARY KEY (document_version_id, page_no)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    section_type TEXT NOT NULL,
    section_title TEXT,
    heading_path TEXT,
    chunk_text TEXT NOT NULL,
    chunk_sha256 CHAR(64) NOT NULL,
    token_count INT NOT NULL,
    embedding VECTOR(3072),
    embedding_model TEXT,
    embedding_created_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE(document_version_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY,
    user_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_session_state (
    session_id UUID PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    active_document_ids BIGINT[] NOT NULL DEFAULT '{}',
    active_record_keys TEXT[] NOT NULL DEFAULT '{}',
    active_entities TEXT[] NOT NULL DEFAULT '{}',
    active_bucket_names TEXT[] NOT NULL DEFAULT '{}',
    last_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    last_citation_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    grounded_summary TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    retrieval_id BIGSERIAL PRIMARY KEY,
    session_id UUID,
    user_query TEXT NOT NULL,
    router_mode TEXT NOT NULL CHECK (router_mode IN ('db_search','memory_scoped_db_search','direct_llm','abstain')),
    extracted_filters JSONB,
    retrieved_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    reranked_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    final_citation_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    confidence_score NUMERIC(5,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_bucket ON source_documents(bucket_name);
CREATE INDEX IF NOT EXISTS idx_doc_versions_doc_id ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_versions_order_date ON document_versions(order_date);
CREATE INDEX IF NOT EXISTS idx_doc_versions_title_trgm ON document_versions USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_version ON document_chunks(document_version_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON document_chunks(section_type);
CREATE INDEX IF NOT EXISTS idx_chunks_pages ON document_chunks(page_start, page_end);
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING GIN (to_tsvector('english', chunk_text));
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = 'idx_chunks_embedding'
    ) THEN
        BEGIN
            CREATE INDEX idx_chunks_embedding
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        EXCEPTION
            WHEN program_limit_exceeded THEN
                RAISE NOTICE 'Skipping idx_chunks_embedding: pgvector ANN indexes do not support 3072-d vector columns';
        END;
    END IF;
END $$;
