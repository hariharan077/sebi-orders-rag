CREATE TABLE IF NOT EXISTS sebi_reference_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_sha256 CHAR(64) NOT NULL,
    fetch_status TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    raw_html TEXT,
    error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sebi_reference_snapshots_source_hash
    ON sebi_reference_snapshots(source_type, content_sha256);

CREATE TABLE IF NOT EXISTS sebi_people (
    person_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    snapshot_id BIGINT REFERENCES sebi_reference_snapshots(snapshot_id),
    canonical_name TEXT NOT NULL,
    designation TEXT,
    role_group TEXT,
    department_name TEXT,
    office_name TEXT,
    email TEXT,
    phone TEXT,
    date_of_joining TEXT,
    staff_no TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sebi_people_source_hash
    ON sebi_people(source_type, row_sha256);

CREATE INDEX IF NOT EXISTS idx_sebi_people_canonical_name
    ON sebi_people(canonical_name);

CREATE INDEX IF NOT EXISTS idx_sebi_people_designation
    ON sebi_people(designation);

CREATE INDEX IF NOT EXISTS idx_sebi_people_role_group
    ON sebi_people(role_group);

CREATE INDEX IF NOT EXISTS idx_sebi_people_office_name
    ON sebi_people(office_name);

CREATE INDEX IF NOT EXISTS idx_sebi_people_department_name
    ON sebi_people(department_name);

CREATE TABLE IF NOT EXISTS sebi_offices (
    office_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    snapshot_id BIGINT REFERENCES sebi_reference_snapshots(snapshot_id),
    office_name TEXT NOT NULL,
    office_type TEXT,
    region TEXT,
    address TEXT,
    email TEXT,
    phone TEXT,
    fax TEXT,
    city TEXT,
    state TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sebi_offices_source_hash
    ON sebi_offices(source_type, row_sha256);

CREATE INDEX IF NOT EXISTS idx_sebi_offices_office_name
    ON sebi_offices(office_name);

CREATE INDEX IF NOT EXISTS idx_sebi_offices_region
    ON sebi_offices(region);

CREATE TABLE IF NOT EXISTS sebi_org_structure (
    org_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    snapshot_id BIGINT REFERENCES sebi_reference_snapshots(snapshot_id),
    leader_name TEXT,
    leader_role TEXT,
    department_name TEXT,
    executive_director_name TEXT,
    executive_director_email TEXT,
    executive_director_phone TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    row_sha256 CHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sebi_org_structure_source_hash
    ON sebi_org_structure(source_type, row_sha256);

CREATE INDEX IF NOT EXISTS idx_sebi_org_structure_department_name
    ON sebi_org_structure(department_name);

CREATE INDEX IF NOT EXISTS idx_sebi_org_structure_leader_name
    ON sebi_org_structure(leader_name);

CREATE INDEX IF NOT EXISTS idx_sebi_org_structure_executive_director_name
    ON sebi_org_structure(executive_director_name);
