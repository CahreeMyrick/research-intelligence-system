# KR-2 and KR-3 evaluation

Run from the repository root with the project's environment. This framework uses
actual `QuestionAnsweringService.answer()` calls, recording the retrieval consumed
by generation exactly once per question. It does not substitute reference answers
or gold passages into the QA prompt.

## Workflow

1. Copy `evals/questions.example.json` to `evals/questions.json`. The three examples
   are source-backed starter questions about one article, not a representative
   acceptance benchmark. Expand to 30–50 answerable questions across your PDFs,
   sections, question types and difficulty levels. Keep tuning examples separate.
2. For each question, supply an expected answer and one or more relevant supporting
   passages with document ID, chunk ID and an exact source quote. Each labelled
   passage should support answering the question; include alternative valid passages
   to avoid false retrieval failures. IDs/text are in `ingestion/corpus/chunks`.
3. Review the questions, answers and passage relevance manually, then set `reviewed`
   to true and freeze a benchmark version. Review judge decisions against human
   grading too; a local 4B judge is not automatically a reliable evaluator.
4. Validate labels without connecting to Postgres, Ollama or DeepEval:

   ```sh
   .venv/bin/python -m evals.run --dataset evals/questions.json --validate-only
   ```

5. Start the same PostgreSQL index and Ollama service used for QA. Ensure the database
   was indexed from the chunk exports used for validation. Run:

   ```sh
   .venv/bin/python -m evals.run \
     --dataset evals/questions.json \
     --top-k 10 --candidate-k 50 \
     --model qwen3:4b-docmind --judge-model qwen3:4b-docmind \
     --output evals/results/run-001.json
   ```

   To try the starter set first, use `--dataset evals/questions.example.json`.
   Generation plus three judge metrics per question can take several minutes with
   local models. `--config` selects indexer configuration; `--ollama-url` selects
   the local model endpoint. Defaults match the QA CLI. This source-tree command
   does not require changing the project's package installation.

6. Read the adjacent Markdown summary and JSON details. The JSON retains question
   labels, exact evidence, generated answers, citations, metric scores, thresholds,
   judge reasons and errors. It includes dataset/corpus hashes, model names,
   embedding version and DeepEval version. Preserve the database snapshot and model
   artifacts with the report for reproduction: a model tag can change, and the local
   corpus hash does not fingerprint the database itself.

## Acceptance rules (rubric v1)

Both KRs use the full benchmark question count as their denominator. The acceptance
threshold is 90% of questions passing, not an average metric score of 0.90.

- **KR-2:** at least one top-k retrieved passage matches a labelled document/chunk
  pair and contains its supporting quote (whitespace normalized). No LLM judge is
  needed. Inspect misses for incomplete gold labels before freezing the next dataset
  version; do not silently change labels halfway through a reported run.
- **KR-3:** a nonempty answer with valid inline `[Evidence N]` references and matching
  structured citations; DeepEval Faithfulness in strict mode (1.0); Answer Relevancy
  at least 0.8; and strict G-Eval cited-answer support (1.0). The latter requires an
  answer covering the reference answer's essentials and supporting citations for all
  substantive factual claims. Support in a different, uncited passage does not count.
  Citation provenance/quotes are checked in Python; semantic claim support is judged.

`--relevancy-threshold` changes the per-answer relevance threshold, and is recorded
in the report. Freeze it along with top-k before acceptance evaluation. Faithfulness
and citation support remain strict. Retrieval uses the QA service's standard hybrid
weights (0.7 semantic / 0.3 lexical), three results per document, and no gold-derived
metadata filters.

Retrieval failures fail both KRs. Generation failures fail KR-3 while preserving the
measured KR-2 outcome. Judge failures fail KR-3 and remain visible as errors. KR-3 is
not conditioned on KR-2 passing. Abstentions fail on this answerable benchmark; test
unanswerable questions separately under a separately defined abstention metric.

The report gives `met` or `not met` against the target and exact counts/rates to show
partial progress. Unreviewed datasets are explicitly provisional and cannot establish
OKR achievement. Exit code 0 means both numerical targets passed, 1 means at least
one missed, and 2 means setup/validation failed. The reviewed flag must also be
checked before using a report as acceptance evidence.

## Tests

```sh
.venv/bin/python -m pytest tests/evals tests/qa/test_qa.py -q
```

These are offline evaluator/pipeline tests, not measurements of corpus quality.
`evals/test_rag.py` remains the original manual DeepEval smoke example.

DeepEval references: [Faithfulness](https://deepeval.com/docs/metrics-faithfulness),
[Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy),
[G-Eval](https://deepeval.com/docs/metrics-llm-evals).
