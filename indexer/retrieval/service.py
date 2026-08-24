from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from indexer.retrieval.models import (
    RetrievalQuery,
    SearchResponse,
    SearchResult,
)
from indexer.retrieval.repository import RetrievalRepository


@dataclass(frozen=True)
class EvidenceItem:
    """
    A single grounded evidence item supplied to an LLM.

    The source text is preserved verbatim. Metadata exists so the model
    can identify where the evidence came from.
    """

    evidence_id: str
    document_id: str
    document_title: Optional[str]
    publication_year: Optional[int]
    section_title: str
    section_path: list[str]
    chunk_id: str
    source_text: str
    page_start: Optional[int]
    page_end: Optional[int]
    relevance_score: float


@dataclass
class EvidenceContext:
    """
    LLM-ready retrieval context.

    This object deliberately contains evidence only. It does not contain
    generated claims, summaries, or answers.
    """

    query: str
    items: list[EvidenceItem] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.items

    def to_prompt_context(self) -> str:
        """
        Convert retrieved evidence into a deterministic text block for
        an LLM prompt.

        Source text is not rewritten or summarized.
        """
        if not self.items:
            return "No relevant evidence was retrieved."

        blocks: list[str] = []

        for index, item in enumerate(self.items, start=1):
            location = item.section_title

            if item.page_start is not None:
                if item.page_end is not None and item.page_end != item.page_start:
                    location += (
                        f", pages {item.page_start}-{item.page_end}"
                    )
                else:
                    location += f", page {item.page_start}"

            blocks.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"Document: {item.document_title or item.document_id}",
                        f"Document ID: {item.document_id}",
                        f"Section: {location}",
                        f"Chunk ID: {item.chunk_id}",
                        f"Relevance: {item.relevance_score:.4f}",
                        "",
                        item.source_text,
                    ]
                )
            )

        return "\n\n".join(blocks)


class RetrievalService:
    """
    Application-level retrieval service.

    This is the boundary between the retrieval infrastructure and the
    future LLM layer.

    The service does not call an LLM and does not generate answers.
    """

    def __init__(self, repository: RetrievalRepository):
        self.repository = repository

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 10,
        candidate_k: int = 50,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        max_results_per_document: int = 3,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_id: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
    ) -> EvidenceContext:
        """
        Retrieve grounded evidence for a user's question.

        The user's question is passed directly into the hybrid retrieval
        pipeline. Metadata constraints remain explicit parameters rather
        than being hidden inside the database layer.
        """
        query = RetrievalQuery(
            text=question,
            top_k=top_k,
            candidate_k=candidate_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            max_results_per_document=max_results_per_document,
            year_from=year_from,
            year_to=year_to,
            document_id=document_id,
            doi=doi,
            arxiv_id=arxiv_id,
        )

        response = self.repository.search(query)

        return self._build_evidence_context(response)

    @staticmethod
    def _build_evidence_context(
        response: SearchResponse,
    ) -> EvidenceContext:
        items = [
            RetrievalService._result_to_evidence(
                result,
                index=index,
            )
            for index, result in enumerate(response.results, start=1)
        ]

        return EvidenceContext(
            query=response.original_query,
            items=items,
        )

    @staticmethod
    def _result_to_evidence(
        result: SearchResult,
        *,
        index: int,
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=f"evidence_{index}",
            document_id=result.document_id,
            document_title=result.document_title,
            publication_year=result.publication_year,
            section_title=result.section_title,
            section_path=result.section_path,
            chunk_id=result.chunk_id,
            source_text=result.source_text,
            page_start=result.page_start,
            page_end=result.page_end,
            relevance_score=result.score,
        )
