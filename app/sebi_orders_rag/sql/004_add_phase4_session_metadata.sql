ALTER TABLE retrieval_logs
    ADD COLUMN IF NOT EXISTS query_intent TEXT;

ALTER TABLE retrieval_logs
    ADD COLUMN IF NOT EXISTS route_mode TEXT;

ALTER TABLE retrieval_logs
    ADD COLUMN IF NOT EXISTS answer_confidence NUMERIC(5,4);

ALTER TABLE retrieval_logs
    ADD COLUMN IF NOT EXISTS answer_status TEXT;

ALTER TABLE retrieval_logs
    ADD COLUMN IF NOT EXISTS cited_record_keys TEXT[] DEFAULT '{}';

UPDATE retrieval_logs
SET route_mode = CASE router_mode
    WHEN 'db_search' THEN 'hierarchical_rag'
    WHEN 'memory_scoped_db_search' THEN 'memory_scoped_rag'
    WHEN 'direct_llm' THEN 'direct_llm'
    WHEN 'abstain' THEN 'abstain'
    ELSE route_mode
END
WHERE route_mode IS NULL;

UPDATE retrieval_logs
SET cited_record_keys = '{}'
WHERE cited_record_keys IS NULL;

ALTER TABLE retrieval_logs
    ALTER COLUMN cited_record_keys SET DEFAULT '{}';

ALTER TABLE retrieval_logs
    ALTER COLUMN cited_record_keys SET NOT NULL;

CREATE TABLE IF NOT EXISTS answer_logs (
    answer_id BIGSERIAL PRIMARY KEY,
    session_id UUID,
    user_query TEXT NOT NULL,
    route_mode TEXT NOT NULL,
    query_intent TEXT,
    answer_text TEXT,
    answer_confidence NUMERIC(5,4),
    cited_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    cited_record_keys TEXT[] NOT NULL DEFAULT '{}',
    citation_payload JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_answer_logs_session
        FOREIGN KEY (session_id)
        REFERENCES chat_sessions(session_id)
        ON DELETE SET NULL
);

ALTER TABLE answer_logs
    ADD COLUMN IF NOT EXISTS citation_payload JSONB NOT NULL DEFAULT '[]'::jsonb;

UPDATE answer_logs
SET citation_payload = '[]'::jsonb
WHERE citation_payload IS NULL;

ALTER TABLE answer_logs
    ALTER COLUMN citation_payload SET DEFAULT '[]'::jsonb;

ALTER TABLE answer_logs
    ALTER COLUMN citation_payload SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_answer_logs_session_id
    ON answer_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_answer_logs_created_at
    ON answer_logs(created_at DESC);

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_document_version_ids BIGINT[] DEFAULT '{}';

UPDATE chat_session_state
SET active_document_version_ids = '{}'
WHERE active_document_version_ids IS NULL;

ALTER TABLE chat_session_state
    ALTER COLUMN active_document_version_ids SET DEFAULT '{}';

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_primary_title TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_primary_entity TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_signatory_name TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_signatory_designation TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_order_date DATE;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_order_place TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS active_legal_provisions TEXT[] DEFAULT '{}';

UPDATE chat_session_state
SET active_legal_provisions = '{}'
WHERE active_legal_provisions IS NULL;

ALTER TABLE chat_session_state
    ALTER COLUMN active_legal_provisions SET DEFAULT '{}';

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS current_lookup_family TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS current_lookup_focus TEXT;

ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS current_lookup_query TEXT;
