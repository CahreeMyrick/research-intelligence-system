from __future__ import annotations

import re

from indexer.qa.models import AnswerCitation, AnswerResponse
from indexer.qa.prompt import build_answer_prompt
from indexer.qa.provider import LLMProvider
from indexer.retrieval.service import EvidenceContext, RetrievalService


class QuestionAnsweringService:
    """
    Application-level question answering service.

    Retrieval and generation remain separate responsibilities:

        question
            ↓
        RetrievalService
            ↓
        EvidenceContext
            ↓
        prompt
            ↓
        LLMProvider
            ↓
        AnswerResponse
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider

    def answer(
        self,
        question: str,
        *,
        top_k: int = 10,
        candidate_k: int = 50,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        max_results_per_document: int = 3,
        year_from: int | None = None,
        year_to: int | None = None,
        document_id: str | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
    ) -> AnswerResponse:
        """
        Retrieve evidence and generate a grounded answer.
        """

        evidence = self.retrieval_service.retrieve(
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

        return self.answer_from_evidence(question, evidence)

    def answer_from_evidence(self, question: str, evidence: EvidenceContext) -> AnswerResponse:
        """Generate from the same context returned to the UI, without retrieving twice."""
        if evidence.empty:
            return AnswerResponse(question=question, answer="The indexed corpus contains insufficient evidence to answer this question.")
        prompt = build_answer_prompt(
            question=question,
            evidence=evidence,
        )

        answer_text = self.llm_provider.generate(prompt).strip()
        if not answer_text:
            raise RuntimeError("The answer model returned an empty response")

        citations = self._extract_citations(
            answer_text,
            evidence,
        )

        return AnswerResponse(
            question=question,
            answer=answer_text,
            citations=citations,
        )

    @staticmethod
    def _extract_citations(
        answer: str,
        evidence: EvidenceContext,
    ) -> list[AnswerCitation]:
        """
        Extract [Evidence N] references from the generated answer and map
        them back to the actual retrieved evidence.

        Invalid or fabricated evidence IDs are ignored.
        """

        matches = re.findall(r"\[Evidence\s+(\d+)\]", answer)

        citations: list[AnswerCitation] = []
        seen: set[str] = set()

        for number in matches:
            evidence_id = f"evidence_{number}"

            if evidence_id in seen:
                continue

            item = next(
                (
                    item
                    for item in evidence.items
                    if item.evidence_id == evidence_id
                ),
                None,
            )

            if item is None:
                continue

            citations.append(
                AnswerCitation(
                    evidence_id=item.evidence_id,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    quote=item.source_text,
                )
            )

            seen.add(evidence_id)

        return citations
