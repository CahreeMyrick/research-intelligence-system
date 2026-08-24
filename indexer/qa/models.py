from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    """
    Citation connecting an answer to retrieved source evidence.
    """

    evidence_id: str
    document_id: str
    chunk_id: str
    quote: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    """
    Final response produced by the question-answering layer.

    The answer itself is generated from retrieved evidence, while citations
    preserve the connection back to the underlying corpus.
    """

    question: str
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
