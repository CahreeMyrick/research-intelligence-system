from __future__ import annotations

from indexer.qa.models import AnswerResponse
from indexer.qa.provider import MockLLMProvider
from indexer.qa.service import QuestionAnsweringService
from indexer.retrieval.service import (
    EvidenceContext,
    EvidenceItem,
)


class MockRetrievalService:
    def __init__(self):
        self.questions: list[str] = []

    def retrieve(self, question: str, **kwargs) -> EvidenceContext:
        self.questions.append(question)

        return EvidenceContext(
            query=question,
            items=[
                EvidenceItem(
                    evidence_id="evidence_1",
                    document_id="doc_001",
                    document_title="Test Research Paper",
                    publication_year=2025,
                    section_title="Introduction",
                    section_path=["Introduction"],
                    chunk_id="chunk_001",
                    source_text="Distributed systems use multiple cooperating machines.",
                    page_start=1,
                    page_end=1,
                    relevance_score=0.95,
                ),
            ],
        )


def test_qa_service_returns_answer():
    retrieval = MockRetrievalService()

    llm = MockLLMProvider(
        response=(
            "Distributed systems use multiple cooperating machines. "
            "[Evidence 1]"
        )
    )

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        llm_provider=llm,
    )

    response = service.answer(
        "How do distributed systems work?"
    )

    assert isinstance(response, AnswerResponse)
    assert response.question == "How do distributed systems work?"
    assert "Distributed systems" in response.answer

    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "evidence_1"
    assert response.citations[0].document_id == "doc_001"
    assert response.citations[0].chunk_id == "chunk_001"


def test_qa_service_passes_evidence_to_llm():
    retrieval = MockRetrievalService()
    llm = MockLLMProvider()

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        llm_provider=llm,
    )

    service.answer("What is the architecture?")

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert "What is the architecture?" in prompt
    assert "Distributed systems use multiple cooperating machines." in prompt
    assert "[Evidence 1]" in prompt


def test_fabricated_citation_is_not_returned():
    retrieval = MockRetrievalService()

    llm = MockLLMProvider(
        response=(
            "The answer is supported by [Evidence 1] "
            "and [Evidence 99]."
        )
    )

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        llm_provider=llm,
    )

    response = service.answer("What is supported?")

    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "evidence_1"


def test_no_citations_when_answer_contains_none():
    retrieval = MockRetrievalService()

    llm = MockLLMProvider(
        response="There is insufficient evidence."
    )

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        llm_provider=llm,
    )

    response = service.answer("What happened?")

    assert response.citations == []
