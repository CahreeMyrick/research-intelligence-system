from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ingest.schema import Document as IngestDocument

from indexer.chunking.base import ChunkerConfig
from indexer.chunking.enrichment import (
    build_document_embedding_text,
    build_section_embedding_text,
)
from indexer.chunking.structural import (
    IndexedSectionNode,
    StructuralSemanticChunker,
)
from indexer.chunking.tokenization import RegexFallbackTokenizer
from indexer.citations.resolver import CitationResolver, CorpusCitationCatalog
from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.database.repository import IndexRepository
from indexer.embeddings.base import EmbeddingProvider
from indexer.embeddings.ollama import OllamaEmbeddingProvider
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider
from indexer.pipeline.state import ProcessingDecision, determine_processing_state
from indexer.schemas.chunk import ChunkArtifact
from indexer.schemas.embedding import EmbeddingLevel, EmbeddingRecord
from indexer.schemas.processing import (
    DocumentProcessingRecord,
    ProcessingStatus,
)

logger = logging.getLogger("indexer")


@dataclass
class IndexingSummary:
    discovered: int = 0
    indexed: int = 0
    rechunked: int = 0
    reembedded: int = 0
    citations_changed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_docs: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def format_report(self) -> str:
        lines = [
            f"Documents discovered: {self.discovered}",
            f"Successfully indexed: {self.indexed}",
            f"Skipped (up to date): {self.skipped}",
            f"Failed: {self.failed}",
        ]

        if self.failed_docs:
            lines.append("\nFailed documents:")
            for doc_id, reason in self.failed_docs.items():
                lines.append(f"  {doc_id}: {reason}")

        return "\n".join(lines)


class CorpusIndexer:
    """
    Main indexing pipeline coordinator.

    Discovers parsed JSON documents, manages version-aware incremental
    processing, generates hierarchical chunks, creates embeddings at
    document/section/chunk levels, resolves citations, and commits
    atomically to PostgreSQL.

    Embedding providers:

    - Hugging Face / SentenceTransformers models:
        BAAI/bge-base-en-v1.5

    - Ollama models:
        nomic-embed-text:latest

    Ollama models are detected by the presence of ':' in the model name.
    """

    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        repository: Optional[IndexRepository] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunker: Optional[StructuralSemanticChunker] = None,
    ):
        self.config = config or IndexerConfig.from_env()

        if repository is None:
            db_mgr = DatabaseManager(self.config.database_url)
            self.repository = IndexRepository(db_mgr)
        else:
            self.repository = repository

        # Allow callers/tests to inject a provider directly.
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            self.embedding_provider = self._create_embedding_provider()

        # Ollama does not expose a tokenizer through its embedding API.
        # Use the provider tokenizer when available, otherwise fall back
        # to a deterministic lightweight tokenizer.
        if chunker is None:
            tokenizer = self._get_chunking_tokenizer()

            chunker_config = ChunkerConfig(
                target_tokens=self.config.chunk_target_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
                max_tokens=self.config.chunk_max_tokens,
                chunker_version=self.config.chunker_version,
            )

            self.chunker = StructuralSemanticChunker(
                config=chunker_config,
                tokenizer=tokenizer,
            )
        else:
            self.chunker = chunker

    def _create_embedding_provider(self) -> EmbeddingProvider:
        """
        Create the configured embedding provider.

        Models containing ':' are treated as Ollama models. This handles
        model identifiers such as:

            nomic-embed-text:latest
            mxbai-embed-large:latest

        Everything else is treated as a SentenceTransformers model.
        """

        model_name = self.config.embedding_model

        if ":" in model_name:
            logger.info(
                "Using Ollama embedding provider for model: %s",
                model_name,
            )

            return OllamaEmbeddingProvider(
                model=model_name,
                model_version=self.config.embedding_model_version,
                expected_dimensions=self.config.embedding_dimensions,
            )

        logger.info(
            "Using SentenceTransformers embedding provider for model: %s",
            model_name,
        )

        return SentenceTransformersProvider(
            model_name=model_name,
            model_version=self.config.embedding_model_version,
            expected_dimensions=self.config.embedding_dimensions,
        )

    def _get_chunking_tokenizer(self):
        """
        Return the tokenizer used by the structural chunker.

        SentenceTransformers exposes its Hugging Face tokenizer.

        Ollama's embedding endpoint does not expose a tokenizer, so the
        indexing pipeline uses the deterministic RegexFallbackTokenizer.
        """

        try:
            return self.embedding_provider.tokenizer
        except (NotImplementedError, AttributeError):
            logger.info(
                "Embedding provider %s does not expose a tokenizer; "
                "using RegexFallbackTokenizer for chunking.",
                self.embedding_provider.name,
            )
            return RegexFallbackTokenizer()

    def index_corpus(
        self,
        corpus_dir: str | Path,
        force: bool = False,
    ) -> IndexingSummary:
        start_time = time.time()

        corpus_path = Path(corpus_dir)
        parsed_dir = corpus_path / "parsed"

        if not parsed_dir.exists():
            raise FileNotFoundError(
                f"Parsed documents directory not found: {parsed_dir}"
            )

        # Ensure database schema is created.
        self.repository.init_schema()

        # 1. Discover parsed JSON files.
        json_files = sorted(parsed_dir.glob("*.json"))

        summary = IndexingSummary(
            discovered=len(json_files),
        )

        logger.info(
            "Discovered %d parsed document(s) in %s",
            len(json_files),
            parsed_dir,
        )

        # 2. Validate and load parsed documents.
        loaded_docs: list[IngestDocument] = []

        for jf in json_files:
            try:
                data = json.loads(
                    jf.read_text(encoding="utf-8")
                )

                doc = IngestDocument.model_validate(data)
                loaded_docs.append(doc)

            except Exception as e:
                doc_id = jf.stem

                logger.error(
                    "Failed to parse document JSON %s: %s",
                    jf.name,
                    e,
                )

                summary.failed += 1
                summary.failed_docs[doc_id] = (
                    f"Invalid JSON schema: {e}"
                )

                self.repository.record_document_failure(
                    document_id=doc_id,
                    error_message=f"Invalid JSON schema: {e}",
                    parser_version=self.config.parser_version,
                )

        # 3. Build global citation catalog across the corpus.
        catalog = CorpusCitationCatalog.build_from_documents(
            loaded_docs
        )

        # Also register existing documents already stored in the DB.
        for existing in self.repository.get_corpus_catalog_identifiers():
            catalog.register_document(
                doc_id=existing["id"],
                doi=existing.get("doi"),
                arxiv_id=existing.get("arxiv_id"),
                title=existing.get("title"),
            )

        citation_resolver = CitationResolver(catalog)

        # 4. Retrieve current processing records from database.
        processing_records = (
            self.repository.get_all_processing_records()
        )

        # 5. Process each document atomically with failure isolation.
        for doc in loaded_docs:
            doc_id = doc.identity.id

            current_record = processing_records.get(doc_id)

            decision = determine_processing_state(
                document=doc,
                current_record=current_record,
                chunker_config=self.chunker.config,
                embedding_provider=self.embedding_provider,
                parser_version=self.config.parser_version,
                force=force,
            )

            if (
                decision == ProcessingDecision.UP_TO_DATE
                and not force
            ):
                logger.info(
                    "[%s] Up to date, skipping",
                    doc_id,
                )

                summary.skipped += 1
                continue

            logger.info(
                "[%s] Processing (reason: %s)",
                doc_id,
                decision.value,
            )

            t0 = time.time()

            try:
                # ---------------------------------------------------------
                # Stage A: Chunk document
                # ---------------------------------------------------------
                sections, chunks = self.chunker.chunk_document(doc)

                # Persist reproducible chunk artifact.
                self.chunker.save_chunk_artifact(
                    corpus_path,
                    doc_id,
                    chunks,
                )

                # ---------------------------------------------------------
                # Stage B: Generate embeddings at 3 levels
                # ---------------------------------------------------------
                embeddings = self._generate_embeddings(
                    doc,
                    sections,
                    chunks,
                )

                # ---------------------------------------------------------
                # Stage C: Resolve citations against corpus
                # ---------------------------------------------------------
                citations, unresolved = (
                    citation_resolver.resolve_document_references(
                        source_doc_id=doc_id,
                        references=doc.references,
                    )
                )

                # ---------------------------------------------------------
                # Stage D: Commit document state atomically
                # ---------------------------------------------------------
                proc_record = DocumentProcessingRecord(
                    document_id=doc_id,
                    parser_version=self.config.parser_version,
                    chunker_version=self.chunker.config.chunker_version,
                    chunker_config_hash=self.chunker.config_hash,
                    embedding_model=self.embedding_provider.name,
                    embedding_model_version=self.embedding_provider.version,
                    embedding_dimensions=self.embedding_provider.dimensions,
                    status=ProcessingStatus.COMPLETED,
                )

                self.repository.save_document_atomic(
                    document=doc,
                    sections=sections,
                    chunks=chunks,
                    embeddings=embeddings,
                    citations=citations,
                    unresolved_references=unresolved,
                    processing_record=proc_record,
                )

                elapsed = time.time() - t0

                logger.info(
                    "[%s] Successfully indexed "
                    "(%d sections, %d chunks, %d embeddings, "
                    "%d citations, %d unresolved) in %.2fs",
                    doc_id,
                    len(sections),
                    len(chunks),
                    len(embeddings),
                    len(citations),
                    len(unresolved),
                    elapsed,
                )

                summary.indexed += 1

            except Exception as e:
                logger.error(
                    "[%s] Indexing failed: %s",
                    doc_id,
                    e,
                    exc_info=True,
                )

                summary.failed += 1
                summary.failed_docs[doc_id] = str(e)

                self.repository.record_document_failure(
                    document_id=doc_id,
                    error_message=str(e),
                    parser_version=self.config.parser_version,
                    chunker_version=self.chunker.config.chunker_version,
                    chunker_config_hash=self.chunker.config_hash,
                    embedding_model=self.embedding_provider.name,
                    embedding_model_version=self.embedding_provider.version,
                    embedding_dimensions=self.embedding_provider.dimensions,
                )

        summary.duration_seconds = (
            time.time() - start_time
        )

        return summary

    def _generate_embeddings(
        self,
        doc: IngestDocument,
        sections: list[IndexedSectionNode],
        chunks: list[ChunkArtifact],
    ) -> list[EmbeddingRecord]:
        """
        Generate embedding records at all three retrieval levels:

        1. Document level
        2. Section level
        3. Chunk level
        """

        doc_id = doc.identity.id

        embeddings: list[EmbeddingRecord] = []

        model_name = self.embedding_provider.name
        model_version = self.embedding_provider.version
        dims = self.embedding_provider.dimensions

        # -------------------------------------------------------------
        # 1. Document embedding
        # -------------------------------------------------------------
        section_titles = [
            section.title
            for section in sections
        ]

        doc_emb_text = build_document_embedding_text(
            doc_title=doc.metadata.title,
            abstract=doc.metadata.abstract,
            section_titles=section_titles,
        )

        doc_vector = self.embedding_provider.embed_single(
            doc_emb_text
        )

        # Validate the actual vector dimensions as an additional guard.
        if len(doc_vector) != dims:
            raise ValueError(
                f"Document embedding dimension mismatch: "
                f"provider reports {dims}, "
                f"but returned {len(doc_vector)}."
            )

        embeddings.append(
            EmbeddingRecord(
                id=f"emb_doc_{doc_id}",
                document_id=doc_id,
                section_id=None,
                chunk_id=None,
                level=EmbeddingLevel.DOCUMENT,
                model_name=model_name,
                model_version=model_version,
                dimensions=dims,
                embedding=doc_vector,
            )
        )

        # -------------------------------------------------------------
        # 2. Section embeddings
        # -------------------------------------------------------------
        if sections:
            sec_texts: list[str] = []

            for sec in sections:
                body_text = "\n\n".join(
                    paragraph["text"]
                    for paragraph in sec.paragraphs
                )

                sec_text = build_section_embedding_text(
                    doc_title=doc.metadata.title,
                    section_path=sec.section_path,
                    section_body_text=body_text,
                )

                sec_texts.append(sec_text)

            sec_vectors = self.embedding_provider.embed_texts(
                sec_texts,
                batch_size=self.config.embedding_batch_size,
            )

            if len(sec_vectors) != len(sections):
                raise RuntimeError(
                    f"Embedding provider returned "
                    f"{len(sec_vectors)} section embeddings "
                    f"for {len(sections)} sections."
                )

            for sec, vec in zip(
                sections,
                sec_vectors,
            ):
                if len(vec) != dims:
                    raise ValueError(
                        f"Section embedding dimension mismatch: "
                        f"provider reports {dims}, "
                        f"but returned {len(vec)}."
                    )

                embeddings.append(
                    EmbeddingRecord(
                        id=f"emb_sec_{sec.id}",
                        document_id=None,
                        section_id=sec.id,
                        chunk_id=None,
                        level=EmbeddingLevel.SECTION,
                        model_name=model_name,
                        model_version=model_version,
                        dimensions=dims,
                        embedding=vec,
                    )
                )

        # -------------------------------------------------------------
        # 3. Chunk embeddings
        # -------------------------------------------------------------
        if chunks:
            chunk_texts = [
                chunk.embedding_text
                for chunk in chunks
            ]

            chunk_vectors = self.embedding_provider.embed_texts(
                chunk_texts,
                batch_size=self.config.embedding_batch_size,
            )

            if len(chunk_vectors) != len(chunks):
                raise RuntimeError(
                    f"Embedding provider returned "
                    f"{len(chunk_vectors)} chunk embeddings "
                    f"for {len(chunks)} chunks."
                )

            for chunk, vec in zip(
                chunks,
                chunk_vectors,
            ):
                if len(vec) != dims:
                    raise ValueError(
                        f"Chunk embedding dimension mismatch: "
                        f"provider reports {dims}, "
                        f"but returned {len(vec)}."
                    )

                embeddings.append(
                    EmbeddingRecord(
                        id=f"emb_chk_{chunk.chunk_id}",
                        document_id=None,
                        section_id=None,
                        chunk_id=chunk.chunk_id,
                        level=EmbeddingLevel.CHUNK,
                        model_name=model_name,
                        model_version=model_version,
                        dimensions=dims,
                        embedding=vec,
                    )
                )

        return embeddings
