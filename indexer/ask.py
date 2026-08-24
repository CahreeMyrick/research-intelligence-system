from __future__ import annotations

from dataclasses import dataclass

from indexer.llm.service import AnswerService
from indexer.retrieval.service import EvidenceContext, RetrievalService


@dataclass(frozen=True)
class AskResponse:
    question: str
    answer: str
    evidence: EvidenceContext


class ResearchAssistant:
    """
    End-to-end research question answering.

    Flow:

        question
            ↓
        retrieval
            ↓
        grounded evidence
            ↓
        local Qwen
            ↓
        answer + evidence
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        answering: AnswerService,
    ):
        self.retrieval = retrieval
        self.answering = answering

    def ask(
        self,
        question: str,
        *,
        top_k: int = 8,
        candidate_k: int = 40,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        max_results_per_document: int = 3,
        year_from: int | None = None,
        year_to: int | None = None,
        document_id: str | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
    ) -> AskResponse:
        evidence = self.retrieval.retrieve(
            question,
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

        answer = self.answering.answer(
            question,
            evidence,
        )

        return AskResponse(
            question=question,
            answer=answer,
            evidence=evidence,
        )
