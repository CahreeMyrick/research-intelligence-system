from __future__ import annotations

import json
import shutil
from pathlib import Path
import pytest

from ingest.pipeline import ingest
from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository
from indexer.embeddings.mock import MockEmbeddingProvider
from indexer.pipeline.indexer import CorpusIndexer

TEST_DB_URL = "postgresql://localhost:5432/research_intel"
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = REPO_ROOT / "ingestion" / "papers"


@pytest.fixture
def clean_db():
    db = DatabaseManager(TEST_DB_URL)
    repo = IndexRepository(db)
    repo.init_schema()
    with db.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents CASCADE;")
    return repo


def test_end_to_end_corpus_indexing(tmp_path, clean_db):
    # 1. Ingest papers into temporary corpus directory
    corpus_dir = tmp_path / "corpus"
    ingest(PAPERS_DIR, output_dir=corpus_dir)
    
    assert (corpus_dir / "parsed").is_dir()
    assert (corpus_dir / "manifest.json").is_file()

    # 2. Run indexer
    config = IndexerConfig(database_url=TEST_DB_URL)
    emb_provider = MockEmbeddingProvider(dimensions=768)
    indexer = CorpusIndexer(config=config, repository=clean_db, embedding_provider=emb_provider)

    summary1 = indexer.index_corpus(corpus_dir)

    assert summary1.discovered == 3
    assert summary1.indexed == 3
    assert summary1.skipped == 0
    assert summary1.failed == 0

    # 3. Verify chunk artifacts exist on disk
    chunks_dir = corpus_dir / "chunks"
    assert chunks_dir.is_dir()
    chunk_files = list(chunks_dir.glob("*.json"))
    assert len(chunk_files) == 3

    for cf in chunk_files:
        data = json.loads(cf.read_text(encoding="utf-8"))
        assert "document_id" in data
        assert "chunks" in data
        assert len(data["chunks"]) > 0
        for chk in data["chunks"]:
            assert "source_text" in chk
            assert "embedding_text" in chk
            assert "chunk_id" in chk
            # Vectors must NOT be stored in JSON files
            assert "embedding" not in chk

    # 4. Verify database state
    stats = clean_db.get_stats()
    assert stats["documents"] == 3
    assert stats["sections"] > 0
    assert stats["chunks"] > 0
    assert stats["embeddings"]["document"] == 3
    assert stats["embeddings"]["section"] > 0
    assert stats["embeddings"]["chunk"] > 0

    # 5. Idempotent Second Run: Must skip all documents without re-indexing
    summary2 = indexer.index_corpus(corpus_dir)
    assert summary2.discovered == 3
    assert summary2.indexed == 0
    assert summary2.skipped == 3
    assert summary2.failed == 0

    # 6. Failure isolation test: Put a corrupted JSON file in parsed/
    bad_doc_path = corpus_dir / "parsed" / "doc_bad.json"
    bad_doc_path.write_text("{corrupt: json}", encoding="utf-8")

    summary3 = indexer.index_corpus(corpus_dir)
    assert summary3.discovered == 4
    assert summary3.failed == 1
    assert "doc_bad" in summary3.failed_docs
    # The valid 3 documents are still skipped / preserved
    assert summary3.skipped == 3
