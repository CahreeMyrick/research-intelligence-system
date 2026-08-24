from __future__ import annotations

from typing import Protocol

from indexer.retrieval.service import EvidenceContext


class LLM(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        ...


class AnswerService:
    """
    Generates grounded answers from retrieved evidence.

    The LLM is never given direct database access. It receives only the
    user's question and the evidence returned by the retrieval layer.
    """

    SYSTEM_PROMPT = """\
You are a research assistant.

Answer the user's question using only the supplied evidence.

Rules:
- Do not invent facts that are not supported by the evidence.
- Cite claims using the supplied evidence markers such as [Evidence 1].
- If the evidence is insufficient, say so clearly.
- Preserve uncertainty when the evidence is uncertain.
- Do not fabricate citations or sources.
"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(
        self,
        question: str,
        evidence: EvidenceContext,
    ) -> str:
        """
        Generate a grounded answer from retrieved evidence.
        """

        if evidence.empty:
            return (
                "I could not find relevant evidence to answer that question."
            )

        prompt = self._build_prompt(
            question=question,
            evidence=evidence,
        )

        return self.llm.generate(
            prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.0,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        evidence: EvidenceContext,
    ) -> str:
        evidence_text = evidence.to_prompt_context()

        return f"""\
User question:

{question}

Retrieved evidence:

{evidence_text}

Answer the question using only the retrieved evidence.
Support factual claims with evidence markers such as [Evidence 1].
"""
