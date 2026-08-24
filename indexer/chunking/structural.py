from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ingest.schema import Document as IngestDocument, Section as IngestSection
from indexer.chunking.base import ChunkerConfig
from indexer.chunking.enrichment import build_chunk_embedding_text
from indexer.chunking.tokenization import TokenizerProtocol, RegexFallbackTokenizer
from indexer.schemas.chunk import ChunkArtifact, DocumentChunksArtifact

logger = logging.getLogger(__name__)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class IndexedSectionNode:
    """Internal flattened representation of a section preserving hierarchical metadata."""
    id: str
    document_id: str
    parent_section_id: Optional[str]
    title: str
    level: int
    page_start: Optional[int]
    page_end: Optional[int]
    section_order: int
    section_path: list[str]
    paragraphs: list[dict]  # list of {'text': str, 'page': int, 'position': int}


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving content."""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences if sentences else [text.strip()]


def _split_oversized_sentence(
    sentence: str,
    tokenizer: TokenizerProtocol,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Deterministic token-based fallback splitting for sentences that exceed target_tokens.
    """
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) <= target_tokens:
        return [sentence]

    pieces: list[str] = []
    step = max(1, target_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        decoded = tokenizer.decode(chunk_tokens)
        if decoded:
            pieces.append(decoded)
        else:
            # Fallback for mock/regex tokenizers that don't decode
            words = sentence.split()
            w_start = int(len(words) * (start / len(tokens)))
            w_end = int(len(words) * (end / len(tokens)))
            pieces.append(" ".join(words[w_start:w_end]))
        if end >= len(tokens):
            break
        start += step

    return [p.strip() for p in pieces if p.strip()]


def extract_section_nodes(
    document: IngestDocument,
) -> list[IndexedSectionNode]:
    """
    Recursively traverse document section tree and produce ordered IndexedSectionNodes
    with assigned deterministic section IDs and breadcrumb paths.
    """
    nodes: list[IndexedSectionNode] = []
    doc_id = document.identity.id
    order = 0

    def traverse(sections: list[IngestSection], parent_id: Optional[str], path: list[str]):
        nonlocal order
        for sec in sections:
            current_order = order
            order += 1
            sec_id = f"sec_{doc_id}_{current_order:04d}"
            current_path = path + [sec.title]
            
            p_list = [{"text": p.text, "page": p.page, "position": p.position} for p in sec.paragraphs]
            
            # Infer start and end pages from paragraphs if not explicitly set
            p_pages = [p.page for p in sec.paragraphs if p.page is not None]
            page_start = sec.page_start if sec.page_start is not None else (min(p_pages) if p_pages else None)
            page_end = sec.page_end if sec.page_end is not None else (max(p_pages) if p_pages else page_start)

            node = IndexedSectionNode(
                id=sec_id,
                document_id=doc_id,
                parent_section_id=parent_id,
                title=sec.title,
                level=sec.level,
                page_start=page_start,
                page_end=page_end,
                section_order=current_order,
                section_path=current_path,
                paragraphs=p_list,
            )
            nodes.append(node)

            if sec.subsections:
                traverse(sec.subsections, sec_id, current_path)

    traverse(document.sections, None, [])
    return nodes


class StructuralSemanticChunker:
    """
    Structure-aware semantic chunker that respects section boundaries,
    accumulates paragraphs, splits oversized paragraphs on sentence boundaries
    (with token fallback), and provides within-section overlap.
    """
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        tokenizer: Optional[TokenizerProtocol] = None,
    ):
        self.config = config or ChunkerConfig()
        self.tokenizer = tokenizer or RegexFallbackTokenizer()
        self.config_hash = self.config.config_hash()

    def chunk_document(
        self,
        document: IngestDocument,
    ) -> tuple[list[IndexedSectionNode], list[ChunkArtifact]]:
        """
        Produce sections and chunk artifacts for an ingested document.
        """
        doc_id = document.identity.id
        doc_title = document.metadata.title
        section_nodes = extract_section_nodes(document)
        all_chunks: list[ChunkArtifact] = []
        global_chunk_index = 0

        for sec in section_nodes:
            sec_chunks = self._chunk_section(doc_id, doc_title, sec, global_chunk_index)
            all_chunks.extend(sec_chunks)
            global_chunk_index += len(sec_chunks)

        return section_nodes, all_chunks

    def _chunk_section(
        self,
        doc_id: str,
        doc_title: Optional[str],
        sec: IndexedSectionNode,
        start_chunk_index: int,
    ) -> list[ChunkArtifact]:
        """
        Chunk a single section. Always produces at least one chunk if the section exists,
        even if the section has few tokens (or is empty heading).
        """
        if not sec.paragraphs:
            # Short / empty body section (e.g. container heading)
            source_text = sec.title
            token_count = self.tokenizer.count_tokens(source_text)
            embedding_text = build_chunk_embedding_text(doc_title, sec.section_path, source_text)
            chunk_id = f"chunk_{doc_id}_{start_chunk_index:04d}"

            return [
                ChunkArtifact(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    section_id=sec.id,
                    section_title=sec.title,
                    section_path=sec.section_path,
                    source_text=source_text,
                    embedding_text=embedding_text,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    paragraph_positions=[],
                    token_count=token_count,
                    chunk_index=start_chunk_index,
                    chunker_version=self.config.chunker_version,
                    chunker_config_hash=self.config_hash,
                )
            ]

        # Break paragraphs down into semantic atomic units (paragraphs or sentences for oversized)
        atomic_units: list[dict] = []
        for p in sec.paragraphs:
            p_text = p["text"].strip()
            if not p_text:
                continue
            p_tokens = self.tokenizer.count_tokens(p_text)
            if p_tokens <= self.config.target_tokens:
                atomic_units.append({
                    "text": p_text,
                    "tokens": p_tokens,
                    "page": p["page"],
                    "position": p["position"],
                })
            else:
                # Oversized paragraph: split at sentence boundaries
                sentences = _split_into_sentences(p_text)
                for s in sentences:
                    s_tokens = self.tokenizer.count_tokens(s)
                    if s_tokens <= self.config.target_tokens:
                        atomic_units.append({
                            "text": s,
                            "tokens": s_tokens,
                            "page": p["page"],
                            "position": p["position"],
                        })
                    else:
                        # Sentence itself is oversized: token-based fallback
                        pieces = _split_oversized_sentence(
                            s, self.tokenizer, self.config.target_tokens, self.config.overlap_tokens
                        )
                        for piece in pieces:
                            atomic_units.append({
                                "text": piece,
                                "tokens": self.tokenizer.count_tokens(piece),
                                "page": p["page"],
                                "position": p["position"],
                            })

        if not atomic_units:
            # Fallback if text was whitespace only
            source_text = sec.title
            return [
                ChunkArtifact(
                    chunk_id=f"chunk_{doc_id}_{start_chunk_index:04d}",
                    document_id=doc_id,
                    section_id=sec.id,
                    section_title=sec.title,
                    section_path=sec.section_path,
                    source_text=source_text,
                    embedding_text=build_chunk_embedding_text(doc_title, sec.section_path, source_text),
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    paragraph_positions=[],
                    token_count=self.tokenizer.count_tokens(source_text),
                    chunk_index=start_chunk_index,
                    chunker_version=self.config.chunker_version,
                    chunker_config_hash=self.config_hash,
                )
            ]

        # Accumulate atomic units into chunks with within-section overlap
        chunks: list[ChunkArtifact] = []
        current_units: list[dict] = []
        current_tokens = 0
        chunk_idx = start_chunk_index
        unit_idx = 0

        while unit_idx < len(atomic_units):
            unit = atomic_units[unit_idx]
            
            if current_units and (current_tokens + unit["tokens"] > self.config.target_tokens):
                # Emit chunk
                chunk = self._create_chunk_artifact(
                    doc_id, doc_title, sec, current_units, chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Calculate overlap units from tail of current chunk
                overlap_units: list[dict] = []
                overlap_token_accum = 0
                for u in reversed(current_units):
                    if overlap_token_accum + u["tokens"] <= self.config.overlap_tokens:
                        overlap_units.insert(0, u)
                        overlap_token_accum += u["tokens"]
                    else:
                        break

                current_units = list(overlap_units)
                current_tokens = sum(u["tokens"] for u in current_units)

            current_units.append(unit)
            current_tokens += unit["tokens"]
            unit_idx += 1

        # Emit trailing chunk
        if current_units:
            chunk = self._create_chunk_artifact(
                doc_id, doc_title, sec, current_units, chunk_idx
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_artifact(
        self,
        doc_id: str,
        doc_title: Optional[str],
        sec: IndexedSectionNode,
        units: list[dict],
        chunk_index: int,
    ) -> ChunkArtifact:
        source_text = "\n\n".join(u["text"] for u in units).strip()
        pages = [u["page"] for u in units if u.get("page") is not None]
        positions = sorted(list({u["position"] for u in units if u.get("position") is not None}))
        
        page_start = min(pages) if pages else sec.page_start
        page_end = max(pages) if pages else sec.page_end
        token_count = self.tokenizer.count_tokens(source_text)
        embedding_text = build_chunk_embedding_text(doc_title, sec.section_path, source_text)

        chunk_id = f"chunk_{doc_id}_{chunk_index:04d}"

        return ChunkArtifact(
            chunk_id=chunk_id,
            document_id=doc_id,
            section_id=sec.id,
            section_title=sec.title,
            section_path=sec.section_path,
            source_text=source_text,
            embedding_text=embedding_text,
            page_start=page_start,
            page_end=page_end,
            paragraph_positions=positions,
            token_count=token_count,
            chunk_index=chunk_index,
            chunker_version=self.config.chunker_version,
            chunker_config_hash=self.config_hash,
        )

    def save_chunk_artifact(
        self,
        corpus_dir: str | Path,
        document_id: str,
        chunks: list[ChunkArtifact],
    ) -> Path:
        """
        Save reproducible chunk artifact JSON file to corpus/chunks/<document_id>.json.
        """
        chunks_dir = Path(corpus_dir) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        artifact = DocumentChunksArtifact(
            document_id=document_id,
            chunk_count=len(chunks),
            chunker_version=self.config.chunker_version,
            chunker_config_hash=self.config_hash,
            chunks=chunks,
        )

        out_path = chunks_dir / f"{document_id}.json"
        out_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        return out_path
