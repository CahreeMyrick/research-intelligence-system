from __future__ import annotations

from indexer.llm.service import AnswerService
from indexer.retrieval.service import EvidenceContext, EvidenceItem


class FakeLLM:
    def __init__(self):
        self.prompt = None
        self.system = None

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.prompt = prompt
        self.system = system
        return "The answer is supported by [Evidence 1]."


def test_answer_service_uses_retrieved_evidence():
    llm = FakeLLM()
    service = AnswerService(llm=llm)

    evidence = EvidenceContext(
        query="What is the architecture?",
        items=[
            EvidenceItem(
                evidence_id="evidence_1",
                document_id="doc1",
                document_title="Test Paper",
                publication_year=2025,
                section_title="Architecture",
                section_path=["Architecture"],
                chunk_id="chunk1",
                source_text="The system uses a distributed architecture.",
                page_start=4,
                page_end=4,
                relevance_score=0.95,
            )
        ],
    )

    answer = service.answer(
        "What architecture does the system use?",
        evidence,
    )

    assert "[Evidence 1]" in answer
    assert llm.prompt is not None
    assert "distributed architecture" in llm.prompt
    assert "What architecture does the system use?" in llm.prompt


def test_answer_service_handles_empty_evidence():
    llm = FakeLLM()
    service = AnswerService(llm=llm)

    evidence = EvidenceContext(
        query="unknown question",
        items=[],
    )

    answer = service.answer(
        "unknown question",
        evidence,
    )

    assert "could not find relevant evidence" in answer
    assert llm.prompt is None
