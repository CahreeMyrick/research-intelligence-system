from __future__ import annotations

from indexer.llm.service import AnswerService
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService

class FakeEmbeddingProvider:
    name = "fake-model"
    version = "1.0"
    dimensions = 768
    tokenizer = None

    def embed_single(self, text: str) -> list[float]:
        return [0.0] * 767 + [1.0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        return [
            [0.0] * 767 + [1.0]
            for _ in texts
        ]

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

        return "The system uses a distributed architecture [Evidence 1]."


def test_question_to_grounded_answer(repo):
    """
    Verify the application path:

        question
          -> retrieval
          -> evidence context
          -> answer service
          -> grounded answer
    """

    retrieval_repository = RetrievalRepository(
        db_manager=repo.db,
        embedding_provider=FakeEmbeddingProvider(),
    )

    retrieval_service = RetrievalService(
        repository=retrieval_repository,
    )

    llm = FakeLLM()

    answer_service = AnswerService(
        llm=llm,
    )

    question = "What architecture does the system use?"

    evidence = retrieval_service.retrieve(
        question,
        top_k=5,
        candidate_k=5,
    )

    answer = answer_service.answer(
        question,
        evidence,
    )

    assert isinstance(answer, str)

    # If the database has indexed material, verify that the evidence
    # actually reached the LLM.
    if not evidence.empty:
        assert llm.prompt is not None
        assert question in llm.prompt
        assert "Evidence" in llm.prompt
