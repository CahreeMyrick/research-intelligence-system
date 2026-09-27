"""HTTP adapter for the existing Research Intelligence System RAG service.

This module deliberately delegates retrieval to the existing
``RetrievalService``/``RetrievalRepository`` stack. It adds no alternate
embedding or ranking implementation.

Run from the research-intelligence-system directory with:
    python -m indexer.http_api
"""

from __future__ import annotations

import json
import os
import threading
from importlib import import_module
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from indexer.qa.ollama import OllamaQwenProvider
from indexer.qa.service import QuestionAnsweringService
from indexer.config import IndexerConfig
from indexer.database.connection import DatabaseManager
from indexer.embeddings.ollama import OllamaEmbeddingProvider
from indexer.embeddings.sentence_transformers import SentenceTransformersProvider
from indexer.retrieval.repository import RetrievalRepository
from indexer.retrieval.service import RetrievalService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "ingestion" / "clinivue_config.yaml"
CONFIG_PATH = Path(os.environ.get("RIS_CONFIG", DEFAULT_CONFIG_PATH))
HOST = "127.0.0.1"
PORT = int(os.environ.get("RIS_PORT", "8010"))


def build_retrieval_service() -> tuple[RetrievalService, IndexerConfig]:
    config = IndexerConfig.from_yaml(CONFIG_PATH)
    database = DatabaseManager(config.database_url)
    if ":" in config.embedding_model:
        provider = OllamaEmbeddingProvider(
            model=config.embedding_model,
            model_version=config.embedding_model_version,
            expected_dimensions=config.embedding_dimensions,
        )
    else:
        provider = SentenceTransformersProvider(
            model_name=config.embedding_model,
            model_version=config.embedding_model_version,
            expected_dimensions=config.embedding_dimensions,
        )
    repository = RetrievalRepository(db_manager=database, embedding_provider=provider)
    return RetrievalService(repository=repository), config


SERVICE: RetrievalService | None = None
CONFIG: IndexerConfig | None = None
QA: QuestionAnsweringService | None = None
SERVICE_LOCK = threading.Lock()


def ensure_services():
    global SERVICE, CONFIG, QA
    with SERVICE_LOCK:
        if QA is not None:
            return
        service, config = build_retrieval_service()
        llm = OllamaQwenProvider(model=config.llm_model,
                                timeout_seconds=float(os.environ.get("RIS_LLM_TIMEOUT", "300")),
                                max_tokens=1024,
                                base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        graph_path = os.environ.get("RIS_KNOWLEDGE_GRAPH", "").strip()
        if graph_path:
            knowledge = import_module("indexer.knowledge-infusion")
            service = knowledge.KnowledgeInfusedRetrievalService(
                service, knowledge.LLMQueryParser(llm), knowledge.load_graph(graph_path))
        SERVICE, CONFIG = service, config
        QA = QuestionAnsweringService(service, llm)



def evidence_response(question: str) -> dict:
    ensure_services()
    context = SERVICE.retrieve(
        question,
        top_k=CONFIG.top_k,
        candidate_k=CONFIG.candidate_k,
    )
    answer = QA.answer_from_evidence(question, context)
    graph = getattr(context, "graph_context", None)
    entities = [] if graph is None else [
        {"name": entity.mention, "type": entity.type, "negated": entity.negated,
         "kg_id": entity.kg_id}
        for entity in graph.normalized.linked.entities
    ]
    paths = [] if graph is None else [
        {**path.model_dump(), "relationships": [edge.relation for edge in path.edges]}
        for path in graph.paths
    ]
    sources = []
    for item in context.items:
        sources.append(
            {
                "id": item.evidence_id,
                "title": item.document_title or item.document_id,
                "publisher": "Research Intelligence System",
                "kind": "Indexed research evidence",
                "summary": item.source_text,
                "passage": item.source_text,
                "detail": f"{item.section_title}; chunk {item.chunk_id}",
                "date": str(item.publication_year or "Unknown publication year"),
                "url": None,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "retrieval_score": item.relevance_score,
                "retrieved": True,
            }
        )
    return {
        "question": question,
        "headline": "Answer grounded in indexed research" if sources else "Insufficient indexed evidence",
        "answer": answer.answer,
        "paragraphs": [p for p in answer.answer.split("\n\n") if p.strip()],
        "citations": [citation.model_dump() for citation in answer.citations],
        "entities": entities,
        "graph_paths": paths,
        "knowledge_graph": {"enabled": graph is not None,
                            "metadata": dict(SERVICE.graph.graph) if graph is not None else {}},
        "source_ids": [source["id"] for source in sources],
        "sources": sources,
        "retrieval": {
            "mode": "knowledge_infused_hybrid" if graph else "research_intelligence_system_hybrid",
            "source_count": len(sources),
            "configuration": CONFIG_PATH.name,
            "embedding_model": CONFIG.embedding_model,
            "embedding_dimensions": CONFIG.embedding_dimensions,
            "embedding_model_version": CONFIG.embedding_model_version,
            "retrieved_count": len(sources),
            "llm_model": CONFIG.llm_model,
        },
        "missing_context": ["Clinical context must be supplied separately"],
        "uncertainty": "Retrieved research evidence is not a diagnosis and may not directly apply to an individual.",
        "safety": "For urgent symptoms, seek appropriate medical care rather than relying on this research retrieval service.",
        "disclaimer": "Evidence retrieved from the existing Research Intelligence System; not clinical advice.",
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"ok": True, "service": "research-intelligence-system", "retrieval": "hybrid", "configuration": CONFIG_PATH.name})
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/qa":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32000:
                raise ValueError("Request body must be between 1 and 32000 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
                raise ValueError("question must be a string")
            question = payload["question"].strip()
            if not question or len(question) > 1500:
                raise ValueError("question must contain 1 to 1500 characters")
        except (ValueError, UnicodeDecodeError) as exc:
            self.send_json(400, {"error": "invalid_request", "message": str(exc)})
            return
        try:
            self.send_json(200, evidence_response(question))
        except Exception as exc:  # keep the adapter's error response inspectable
            self.send_json(502, {"error": "retrieval_failed", "message": str(exc)})


if __name__ == "__main__":
    print(f"Research Intelligence System API: http://{HOST}:{PORT}", file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
