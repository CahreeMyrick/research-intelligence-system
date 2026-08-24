"""
Smoke tests. Run with:  PYTHONPATH=. python -m pytest tests/ -v
(or just:               PYTHONPATH=. python tests/test_pipeline.py)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.pipeline import ingest, ingest_one

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SCRIPT = REPO_ROOT / "scripts" / "make_sample_pdf.py"


def _make_sample_dir(tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / "sample_001.pdf"
    subprocess.run([sys.executable, str(SAMPLE_SCRIPT), str(pdf_path)], check=True)
    return tmp_dir


def test_single_document_structure(tmp_path):
    papers_dir = _make_sample_dir(tmp_path / "papers")
    doc = ingest_one(papers_dir / "sample_001.pdf")

    assert doc.metadata.title == "ViTGAN: Generative Adversarial Networks with Vision Transformers"
    assert doc.metadata.authors == ["Jane Doe", "John Smith", "Alice Wu"]
    assert doc.metadata.arxiv_id == "2107.04589"
    assert doc.metadata.publication_year == 2021
    assert doc.metadata.abstract and doc.metadata.abstract.startswith("We introduce ViTGAN")

    top_titles = [s.title for s in doc.sections]
    assert top_titles == [
        "Abstract",
        "1 Introduction",
        "2 Related Work",
        "3 ViTGAN",
        "4 Experiments",
        "5 Conclusion",
    ]
    related_work = next(s for s in doc.sections if s.title == "2 Related Work")
    assert [sub.title for sub in related_work.subsections] == [
        "2.1 Generative Adversarial Networks",
        "2.2 Vision Transformers",
    ]

    assert len(doc.references) == 3
    assert doc.references[0].year == 2017
    assert doc.references[2].doi == "10.1145/3422622"

    assert doc.warnings == []


def test_identity_is_content_based_not_path_based(tmp_path):
    papers_dir = _make_sample_dir(tmp_path / "papers")
    original = papers_dir / "sample_001.pdf"
    renamed = papers_dir / "totally_different_name.pdf"
    shutil.copyfile(original, renamed)

    doc_a = ingest_one(original)
    doc_b = ingest_one(renamed)

    assert doc_a.identity.id == doc_b.identity.id
    assert doc_a.identity.source.content_hash == doc_b.identity.source.content_hash
    assert doc_a.identity.source.filename != doc_b.identity.source.filename


def test_corpus_deduplicates_identical_files(tmp_path):
    papers_dir = _make_sample_dir(tmp_path / "papers")
    shutil.copyfile(papers_dir / "sample_001.pdf", papers_dir / "sample_001_dupe.pdf")

    corpus = ingest(papers_dir)
    assert len(corpus.documents) == 1  # duplicate content -> deduped


def test_corpus_isolates_a_broken_pdf(tmp_path):
    papers_dir = _make_sample_dir(tmp_path / "papers")
    (papers_dir / "not_actually_a_pdf.pdf").write_bytes(b"this is not a pdf")

    corpus = ingest(papers_dir)
    # the good paper still made it through despite the broken one
    assert len(corpus.documents) == 1
    assert corpus.documents[0].metadata.title is not None


def test_corpus_generates_target_structure(tmp_path):
    papers_dir = _make_sample_dir(tmp_path / "papers")
    corpus_dir = tmp_path / "corpus"

    corpus = ingest(papers_dir, output_dir=corpus_dir)
    assert len(corpus.documents) == 1
    doc = corpus.documents[0]

    # Verify target directory structure
    raw_pdf = corpus_dir / "raw" / "pdf" / f"{doc.identity.id}.pdf"
    parsed_json = corpus_dir / "parsed" / f"{doc.identity.id}.json"
    chunks_dir = corpus_dir / "chunks"
    manifest_json = corpus_dir / "manifest.json"

    assert raw_pdf.exists() and raw_pdf.is_file()
    assert parsed_json.exists() and parsed_json.is_file()
    assert chunks_dir.exists() and chunks_dir.is_dir()
    assert not (corpus_dir / "embeddings").exists()  # Embeddings stored in pgvector only
    assert manifest_json.exists() and manifest_json.is_file()

    # Verify manifest content
    import json
    manifest_data = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest_data["document_count"] == 1
    assert len(manifest_data["documents"]) == 1
    entry = manifest_data["documents"][0]
    assert entry["id"] == doc.identity.id
    assert entry["title"] == doc.metadata.title
    assert entry["raw_path"] == f"raw/pdf/{doc.identity.id}.pdf"
    assert entry["parsed_path"] == f"parsed/{doc.identity.id}.json"

    # Verify loading back from directory
    from ingest.schema import Corpus
    loaded_corpus = Corpus.load(corpus_dir)
    assert len(loaded_corpus.documents) == 1
    assert loaded_corpus.documents[0].identity.id == doc.identity.id
    assert loaded_corpus.documents[0].metadata.title == doc.metadata.title


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_single_document_structure(Path(d) / "t1")
    with tempfile.TemporaryDirectory() as d:
        test_identity_is_content_based_not_path_based(Path(d) / "t2")
    with tempfile.TemporaryDirectory() as d:
        test_corpus_deduplicates_identical_files(Path(d) / "t3")
    with tempfile.TemporaryDirectory() as d:
        test_corpus_isolates_a_broken_pdf(Path(d) / "t4")
    with tempfile.TemporaryDirectory() as d:
        test_corpus_generates_target_structure(Path(d) / "t5")
    print("All smoke tests passed.")

