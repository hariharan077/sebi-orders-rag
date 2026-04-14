CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS embedding_status TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_document_versions_embedding_status'
    ) THEN
        ALTER TABLE document_versions
            ADD CONSTRAINT chk_document_versions_embedding_status
            CHECK (embedding_status IN ('pending','processing','done','failed'));
    END IF;
END $$;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS embedding_error TEXT;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS embedding_dim INT;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS section_key TEXT;

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS chunk_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

DROP INDEX IF EXISTS idx_chunks_embedding;

ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE VECTOR(1536);

CREATE TABLE IF NOT EXISTS document_nodes (
    document_node_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL UNIQUE REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    node_text TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding VECTOR(1536),
    embedding_model TEXT,
    embedding_created_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS section_nodes (
    section_node_id BIGSERIAL PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    section_key TEXT NOT NULL,
    section_type TEXT NOT NULL,
    section_title TEXT,
    heading_path TEXT,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    node_text TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding VECTOR(1536),
    embedding_model TEXT,
    embedding_created_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(document_version_id, section_key)
);

CREATE INDEX IF NOT EXISTS idx_document_versions_embedding_status
    ON document_versions(embedding_status);

CREATE INDEX IF NOT EXISTS idx_chunks_section_key
    ON document_chunks(section_key);

CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
    ON document_chunks USING GIN (chunk_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_document_nodes_doc_version
    ON document_nodes(document_version_id);

CREATE INDEX IF NOT EXISTS idx_document_nodes_fts
    ON document_nodes USING GIN (to_tsvector('english', node_text));

CREATE INDEX IF NOT EXISTS idx_document_nodes_text_trgm
    ON document_nodes USING GIN (node_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_section_nodes_doc_version
    ON section_nodes(document_version_id);

CREATE INDEX IF NOT EXISTS idx_section_nodes_section_type
    ON section_nodes(section_type);

CREATE INDEX IF NOT EXISTS idx_section_nodes_pages
    ON section_nodes(page_start, page_end);

CREATE INDEX IF NOT EXISTS idx_section_nodes_fts
    ON section_nodes USING GIN (to_tsvector('english', node_text));

CREATE INDEX IF NOT EXISTS idx_section_nodes_text_trgm
    ON section_nodes USING GIN (node_text gin_trgm_ops);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = 'idx_document_nodes_embedding'
    ) THEN
        BEGIN
            CREATE INDEX idx_document_nodes_embedding
                ON document_nodes
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50);
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Skipping idx_document_nodes_embedding: %', SQLERRM;
        END;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = 'idx_section_nodes_embedding'
    ) THEN
        BEGIN
            CREATE INDEX idx_section_nodes_embedding
                ON section_nodes
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Skipping idx_section_nodes_embedding: %', SQLERRM;
        END;
    END IF;
END $$;

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
            WHEN OTHERS THEN
                RAISE NOTICE 'Skipping idx_chunks_embedding: %', SQLERRM;
        END;
    END IF;
END $$;
