"""Usage: python -m evals.run --dataset evals/questions.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from evals.framework import Benchmark, RunSettings, run_benchmark, validate_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate corpus QA against KR-2 and KR-3")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("ingestion/corpus/chunks"))
    parser.add_argument("--config", type=Path, default=Path("indexer/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.json"))
    parser.add_argument("--model", default="qwen3:4b-docmind")
    parser.add_argument("--judge-model", default="qwen3:4b-docmind")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--relevancy-threshold", type=float, default=0.8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.dataset.read_bytes()
        benchmark = Benchmark.model_validate_json(raw)
        settings = RunSettings(top_k=args.top_k, candidate_k=args.candidate_k,
                               relevancy_threshold=args.relevancy_threshold)
        corpus_hash = validate_corpus(benchmark, args.corpus)
        if args.validate_only:
            print(f"Valid: {len(benchmark.questions)} questions; reviewed={benchmark.reviewed}")
            return 0

        from indexer.config import IndexerConfig
        from indexer.database.connection import DatabaseManager
        from indexer.embeddings.sentence_transformers import SentenceTransformersProvider
        from indexer.retrieval.repository import RetrievalRepository
        from indexer.retrieval.service import RetrievalService
        from indexer.qa.ollama import OllamaQwenProvider
        from deepeval.models import OllamaModel
        from evals.judge import DeepEvalJudge

        config = IndexerConfig.from_yaml(args.config)
        retrieval = RetrievalService(RetrievalRepository(
            db_manager=DatabaseManager(config.database_url),
            embedding_provider=SentenceTransformersProvider(
                model_name=config.embedding_model, model_version=config.embedding_model_version,
                expected_dimensions=config.embedding_dimensions)))
        provider = OllamaQwenProvider(model=args.model, base_url=args.ollama_url)
        judge = DeepEvalJudge(OllamaModel(model=args.judge_model,
                             base_url=args.ollama_url, temperature=0),
                             settings.relevancy_threshold)
        report = run_benchmark(benchmark, retrieval, provider, judge, settings)
        report["provenance"] = dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_sha256=hashlib.sha256(raw).hexdigest(), corpus_sha256=corpus_hash,
            answer_model=args.model, judge_model=args.judge_model,
            embedding_model=config.embedding_model,
            embedding_model_version=config.embedding_model_version,
            deepeval_version=version("deepeval"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        lines = [f"# Evaluation: {benchmark.name} ({benchmark.version})", "",
                 f"Reviewed benchmark: {benchmark.reviewed}", "",
                 "| KR | Passed | Rate | Status |", "| --- | --- | --- | --- |"]
        for kr, result in report["summary"].items():
            lines.append(f"| {kr.upper()} | {result['passed']}/{result['total']} | "
                         f"{result['rate']:.1%} | {result['status']} |")
        if not benchmark.reviewed:
            lines.extend(["", "Provisional results only: benchmark requires human review."])
        lines.extend(["", "Per-question evidence, scores, reasons and errors are in the JSON report."])
        args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0 if all(r["status"] == "met" for r in report["summary"].values()) else 1
    except Exception as exc:
        print(f"Evaluation setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
