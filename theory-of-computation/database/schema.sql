CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    object_storage_uri TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE pages (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,

    UNIQUE(document_id, page_number)
);

CREATE TABLE sections (
    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    parent_section_id BIGINT
        REFERENCES sections(id)
        ON DELETE CASCADE,

    heading TEXT,
    level INTEGER NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    page_id BIGINT
        REFERENCES pages(id)
        ON DELETE SET NULL,

    section_id BIGINT
        REFERENCES sections(id)
        ON DELETE SET NULL,

    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,

    UNIQUE(document_id, chunk_index)
);
