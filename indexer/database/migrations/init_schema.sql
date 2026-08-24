-- Extension for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. documents
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(128) PRIMARY KEY,
    content_hash VARCHAR(64) NOT NULL,
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    title TEXT,
    abstract TEXT,
    publication_year INTEGER,
    venue TEXT,
    doi VARCHAR(256),
    arxiv_id VARCHAR(128),
    other_identifiers JSONB DEFAULT '{}'::jsonb,
    page_count INTEGER DEFAULT 0,
    parser_version VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_arxiv_id ON documents(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_pub_year ON documents(publication_year);

-- 2. sections
CREATE TABLE IF NOT EXISTS sections (
    id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(128) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_section_id VARCHAR(128) REFERENCES sections(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_order INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent_id ON sections(parent_section_id);
CREATE INDEX IF NOT EXISTS idx_sections_doc_order ON sections(document_id, section_order);

-- 3. chunks
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(128) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id VARCHAR(128) NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    token_count INTEGER NOT NULL,
    chunker_version VARCHAR(64) NOT NULL,
    chunker_config_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_index ON chunks(document_id, chunk_index);

-- 4. embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(128) REFERENCES documents(id) ON DELETE CASCADE,
    section_id VARCHAR(128) REFERENCES sections(id) ON DELETE CASCADE,
    chunk_id VARCHAR(128) REFERENCES chunks(id) ON DELETE CASCADE,
    level VARCHAR(32) NOT NULL CHECK (level IN ('document', 'section', 'chunk')),
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64),
    dimensions INTEGER NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_embedding_target CHECK (
        (level = 'document' AND document_id IS NOT NULL AND section_id IS NULL AND chunk_id IS NULL) OR
        (level = 'section' AND section_id IS NOT NULL AND document_id IS NULL AND chunk_id IS NULL) OR
        (level = 'chunk' AND chunk_id IS NOT NULL AND document_id IS NULL AND section_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON embeddings(document_id) WHERE document_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_embeddings_section ON embeddings(section_id) WHERE section_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id) WHERE chunk_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_embeddings_level ON embeddings(level);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);

-- 5. citations
CREATE TABLE IF NOT EXISTS citations (
    id VARCHAR(128) PRIMARY KEY,
    source_document_id VARCHAR(128) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id VARCHAR(128) REFERENCES documents(id) ON DELETE SET NULL,
    reference_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    doi VARCHAR(256),
    arxiv_id VARCHAR(128),
    resolution_method VARCHAR(64),
    resolution_confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citations_source ON citations(source_document_id);
CREATE INDEX IF NOT EXISTS idx_citations_target ON citations(target_document_id);
CREATE INDEX IF NOT EXISTS idx_citations_doi ON citations(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_citations_arxiv ON citations(arxiv_id) WHERE arxiv_id IS NOT NULL;

-- 6. unresolved_references
CREATE TABLE IF NOT EXISTS unresolved_references (
    id VARCHAR(128) PRIMARY KEY,
    source_document_id VARCHAR(128) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    reference_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    doi VARCHAR(256),
    arxiv_id VARCHAR(128),
    title TEXT,
    authors JSONB DEFAULT '[]'::jsonb,
    year INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_unresolved_source ON unresolved_references(source_document_id);

-- 7. document_processing
CREATE TABLE IF NOT EXISTS document_processing (
    document_id VARCHAR(128) PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    parser_version VARCHAR(64),
    chunker_version VARCHAR(64),
    chunker_config_hash VARCHAR(64),
    embedding_model VARCHAR(128),
    embedding_model_version VARCHAR(64),
    embedding_dimensions INTEGER,
    status VARCHAR(32) NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    last_indexed_at TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_doc_proc_status ON document_processing(status);
