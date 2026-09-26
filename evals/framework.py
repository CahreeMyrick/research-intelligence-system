from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def normalize(text: str) -> str:
    return " ".join(text.split())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SupportingPassage(StrictModel):
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Question(StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    supporting_passages: list[SupportingPassage] = Field(min_length=1)


class Benchmark(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    reviewed: bool = False
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return self


class RunSettings(StrictModel):
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=1, le=500)
    target: float = Field(default=0.9, gt=0, le=1)
    relevancy_threshold: float = Field(default=0.8, gt=0, le=1)


def validate_corpus(benchmark: Benchmark, directory: Path) -> str:
    """Check labels against the frozen local chunk exports and fingerprint them."""
    chunks = {}
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(raw)
        for chunk in json.loads(raw)["chunks"]:
            key = (chunk["document_id"], chunk["chunk_id"])
            if key in chunks:
                raise ValueError(f"Duplicate corpus chunk: {key}")
            chunks[key] = normalize(chunk["source_text"])
    for question in benchmark.questions:
        for passage in question.supporting_passages:
            text = chunks.get((passage.document_id, passage.chunk_id))
            if text is None or normalize(passage.quote) not in text:
                raise ValueError(f"{question.id}: supporting passage missing or stale: {passage.chunk_id}")
    return digest.hexdigest()


class RecordingRetrieval:
    """Capture the single retrieval actually consumed by the QA service."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.evidence = None

    def retrieve(self, *args, **kwargs):
        self.evidence = self.delegate.retrieve(*args, **kwargs)
        return self.evidence


def retrieval_hits(question, evidence):
    return [item.chunk_id for item in evidence.items if any(
        item.document_id == gold.document_id
        and item.chunk_id == gold.chunk_id
        and normalize(gold.quote) in normalize(item.source_text)
        for gold in question.supporting_passages
    )]


def citation_errors(response, evidence):
    items = {item.evidence_id: item for item in evidence.items}
    references = {f"evidence_{n}" for n in re.findall(r"\[Evidence\s+(\d+)\]", response.answer)}
    errors = []
    if not references:
        errors.append("No inline evidence citations")
    for ref in sorted(references - items.keys()):
        errors.append(f"Unknown citation: {ref}")
    citations = {citation.evidence_id: citation for citation in response.citations}
    if citations.keys() != references:
        errors.append("Inline citations and structured citations do not match")
    for ref, citation in citations.items():
        item = items.get(ref)
        if item is None or (citation.document_id, citation.chunk_id) != (item.document_id, item.chunk_id):
            errors.append(f"Invalid citation provenance: {ref}")
        elif normalize(citation.quote) not in normalize(item.source_text):
            errors.append(f"Citation quote is not in source: {ref}")
    return errors


def run_benchmark(benchmark, retrieval, provider, judge, settings):
    from indexer.qa.service import QuestionAnsweringService

    recorder = RecordingRetrieval(retrieval)
    qa = QuestionAnsweringService(recorder, provider)
    rows = []
    for question in benchmark.questions:
        recorder.evidence = None
        row = dict(id=question.id, question=question.question,
                   expected_answer=question.expected_answer,
                   supporting_passages=[p.model_dump() for p in question.supporting_passages],
                   kr2_pass=False, kr3_pass=False, metrics={}, errors=[])
        response = None
        try:
            response = qa.answer(question.question, top_k=settings.top_k,
                                 candidate_k=settings.candidate_k)
            row["response"] = response.model_dump()
        except Exception as exc:
            row["errors"].append(f"Pipeline: {type(exc).__name__}: {exc}")
        evidence = recorder.evidence
        if evidence is not None:
            row["evidence"] = asdict(evidence)
            row["retrieval_hits"] = retrieval_hits(question, evidence)
            row["kr2_pass"] = bool(row["retrieval_hits"])
        if response is not None and evidence is not None:
            row["errors"].extend(citation_errors(response, evidence))
            if not response.answer.strip():
                row["errors"].append("Empty answer")
            try:
                row["metrics"] = judge.grade(question, response, evidence)
                for name, result in row["metrics"].items():
                    if result.get("error"):
                        row["errors"].append(f"Judge {name}: {result['error']}")
                required = {"faithfulness", "answer_relevancy", "citation_support"}
                row["kr3_pass"] = (not row["errors"] and required <= row["metrics"].keys()
                                   and all(row["metrics"][key]["passed"] for key in required))
            except Exception as exc:
                row["errors"].append(f"Judge: {type(exc).__name__}: {exc}")
        rows.append(row)
    summary = {}
    for kr in ("kr2", "kr3"):
        passed = sum(row[f"{kr}_pass"] for row in rows)
        rate = passed / len(rows)
        summary[kr] = dict(passed=passed, total=len(rows), rate=rate,
                           status="met" if rate >= settings.target else "not met")
    return dict(benchmark=benchmark.name, version=benchmark.version,
                reviewed=benchmark.reviewed, settings=settings.model_dump(),
                summary=summary, questions=rows)
