CREATE TABLE IF NOT EXISTS sebi_board_members (
    board_member_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL DEFAULT 'board_members',
    source_url TEXT NOT NULL,
    snapshot_id BIGINT REFERENCES sebi_reference_snapshots(snapshot_id),
    canonical_name TEXT NOT NULL,
    board_role TEXT NOT NULL,
    category TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'board_members';

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS source_url TEXT;

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS snapshot_id BIGINT REFERENCES sebi_reference_snapshots(snapshot_id);

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS canonical_name TEXT;

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS board_role TEXT;

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS category TEXT;

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS row_sha256 CHAR(64);

ALTER TABLE sebi_board_members
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS uq_sebi_board_members_source_hash
    ON sebi_board_members(source_type, row_sha256);

CREATE INDEX IF NOT EXISTS idx_sebi_board_members_canonical_name
    ON sebi_board_members(canonical_name);

CREATE INDEX IF NOT EXISTS idx_sebi_board_members_category
    ON sebi_board_members(category);

CREATE INDEX IF NOT EXISTS idx_sebi_reference_snapshots_source_fetched_at
    ON sebi_reference_snapshots(source_type, fetched_at DESC, snapshot_id DESC);

CREATE OR REPLACE VIEW sebi_latest_reference_snapshots_v AS
SELECT DISTINCT ON (source_type)
    snapshot_id,
    source_type,
    source_url,
    fetched_at,
    content_sha256,
    fetch_status,
    parse_status,
    error
FROM sebi_reference_snapshots
ORDER BY source_type, fetched_at DESC, snapshot_id DESC;

CREATE OR REPLACE VIEW sebi_active_board_members_v AS
SELECT
    board_member_id,
    source_type,
    source_url,
    snapshot_id,
    canonical_name,
    board_role,
    category,
    is_active,
    row_sha256,
    updated_at
FROM sebi_board_members
WHERE is_active = TRUE;
