ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS chunking_version TEXT;

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS chunk_count INT;
