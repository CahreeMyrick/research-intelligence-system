from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """
    Provider interface for answer generation.

    Keeping this as a protocol allows the QA layer to be tested without
    making real LLM API calls.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...


class MockLLMProvider:
    """
    Deterministic LLM implementation for tests.

    Production providers can implement the same interface later.
    """

    def __init__(
        self,
        response: str = "Mock grounded answer.",
        name: str = "mock-llm",
        version: str = "1.0",
    ):
        self.response = response
        self._name = name
        self._version = version
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response
