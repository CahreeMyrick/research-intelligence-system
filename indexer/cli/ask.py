from __future__ import annotations

import argparse

from indexer.database.connection import DatabaseManager
from indexer.embeddings.ollama import OllamaEmbeddingProvider
from indexer.llm import AnswerService, OllamaLLM
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService


DATABASE_URL = "postgresql://localhost:5432/research_intel"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about the indexed research corpus."
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask.",
    )

    args = parser.parse_args()

    question = " ".join(args.question)

    db = DatabaseManager(DATABASE_URL)

    embedding_provider = OllamaEmbeddingProvider(
        model="nomic-embed-text:latest",
    )

    retrieval_repository = RetrievalRepository(
        db_manager=db,
        embedding_provider=embedding_provider,
    )

    retrieval_service = RetrievalService(
        repository=retrieval_repository,
    )

    llm = OllamaLLM(
        model="qwen3:4b-docmind",
    )

    answer_service = AnswerService(
        llm=llm,
    )

    evidence = retrieval_service.retrieve(
        question,
        top_k=5,
        candidate_k=20,
    )

    answer = answer_service.answer(
        question,
        evidence,
    )

    print()
    print("ANSWER")
    print("------")
    print(answer)

    print()
    print(f"Evidence retrieved: {len(evidence.items)}")


if __name__ == "__main__":
    main()
