CREATE TABLE IF NOT EXISTS sebi_canonical_people (
    canonical_person_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name_key TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    designation TEXT,
    designation_group TEXT NOT NULL,
    department_name TEXT,
    department_aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    office_name TEXT,
    office_city TEXT,
    office_region TEXT,
    office_aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    email TEXT,
    phone TEXT,
    date_of_joining TEXT,
    staff_no TEXT,
    role_group TEXT,
    board_role TEXT,
    board_category TEXT,
    source_priority INTEGER NOT NULL DEFAULT 0,
    active_status BOOLEAN NOT NULL DEFAULT TRUE,
    source_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_row_count INTEGER NOT NULL DEFAULT 0,
    merge_notes JSONB NOT NULL DEFAULT '[]'::jsonb,
    merged_row_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE sebi_canonical_people
    ADD COLUMN IF NOT EXISTS staff_no TEXT;

ALTER TABLE sebi_canonical_people
    ADD COLUMN IF NOT EXISTS merge_notes JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE sebi_canonical_people
    ADD COLUMN IF NOT EXISTS merged_row_keys JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_people_name
    ON sebi_canonical_people (canonical_name);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_people_name_key
    ON sebi_canonical_people (normalized_name_key);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_people_designation_group
    ON sebi_canonical_people (designation_group);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_people_office_name
    ON sebi_canonical_people (office_name);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_people_staff_no
    ON sebi_canonical_people (staff_no);

CREATE TABLE IF NOT EXISTS sebi_canonical_offices (
    canonical_office_id TEXT PRIMARY KEY,
    office_name TEXT NOT NULL,
    normalized_office_key TEXT NOT NULL,
    office_aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    office_type TEXT,
    city TEXT,
    state TEXT,
    region TEXT,
    address TEXT,
    email TEXT,
    phone TEXT,
    fax TEXT,
    source_priority INTEGER NOT NULL DEFAULT 0,
    active_status BOOLEAN NOT NULL DEFAULT TRUE,
    source_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_row_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_offices_name
    ON sebi_canonical_offices (office_name);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_offices_city
    ON sebi_canonical_offices (city);

CREATE INDEX IF NOT EXISTS idx_sebi_canonical_offices_region
    ON sebi_canonical_offices (region);

CREATE TABLE IF NOT EXISTS sebi_designation_counts (
    designation_group TEXT PRIMARY KEY,
    designation_label TEXT NOT NULL,
    people_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sebi_office_counts (
    canonical_office_id TEXT PRIMARY KEY,
    office_name TEXT NOT NULL,
    city TEXT,
    region TEXT,
    people_count INTEGER NOT NULL DEFAULT 0,
    office_count INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sebi_department_counts (
    department_name TEXT PRIMARY KEY,
    people_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sebi_role_counts (
    role_key TEXT PRIMARY KEY,
    role_label TEXT NOT NULL,
    people_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
