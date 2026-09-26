"""Manual DeepEval smoke example; use evals.run for corpus evaluation."""

from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


def main():
    judge = OllamaModel(
        model="qwen3:4b-docmind",
        base_url="http://localhost:11434",
        temperature=0,
    )

    test_case = LLMTestCase(
        input="What is retrieval augmented generation?",
        actual_output=(
            "Retrieval augmented generation combines retrieval "
            "with language model generation."
        ),
        retrieval_context=[
            """
            Retrieval-augmented generation retrieves relevant
            documents and supplies them to a language model
            before answer generation.
            """
        ],
    )

    metrics = [
        AnswerRelevancyMetric(model=judge),
        FaithfulnessMetric(model=judge),
    ]

    evaluate(
        test_cases=[test_case],
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
