# Connected Clarivue app

From this directory, start the UI, proxy, and research API together:

```sh
.venv/bin/python -m indexer.web_app
```

Open http://127.0.0.1:8002. Defaults: UI port 8002, internal backend port 8012.
Use `--port` and `--backend-port` to override. The sibling `../clarivue_qa`
checkout supplies the web assets and proxy (`--web-root` overrides its location).

Every submitted question, including the example questions, follows this path:

1. Browser posts `/api/qa` to the same-origin Python web proxy.
2. The research API uses the embedding model and PostgreSQL/pgvector index from `ingestion/clinivue_config.yaml`.
3. When a graph is supplied, the knowledge adapter extracts concepts, links them, discovers paths, and expands retrieval queries.
4. The QA service sends the retrieved passages to the configured Ollama model.
5. The UI renders the generated answer, validated citation links, source passages, and graph associations from that response.

Graph enrichment requires an explicit graph JSON:

```sh
.venv/bin/python -m indexer.web_app --knowledge-graph /path/to/curated-graph.json
```

To exercise the existing demo graph, use
`--knowledge-graph indexer/knowledge-infusion/demo_graph.json`. Its associations
are unverified demo data, not clinical evidence. It contains chest X-ray concepts,
not angina/running concepts. Graph edges guide search; only retrieved passages
support the answer. Without a supplied graph, hybrid retrieval and generation
still work and the UI reports graph enrichment as unconfigured.

Requirements: installed project dependencies (including `graph` extra for graphs),
running medical PostgreSQL (`clinivue` on port 5433) with the indexed `ingestion/medical_corpus`, cached or downloadable configured
embedding model, and Ollama with the configured `llm_model`. Override configuration
with `--config`. The default is `ingestion/clinivue_config.yaml`, using
`BAAI/bge-base-en-v1.5` 768-dimensional vectors. The unrelated
`indexer/config.yaml` / `research_intel` database is used only if explicitly selected. `OLLAMA_HOST` sets the Ollama base URL. A corpus without relevant
medical evidence cannot support medical answers.

For separate processes, `python -m indexer.http_api` defaults to port 8010
(`RIS_PORT` overrides it). `python ../clarivue_qa/api_server.py` defaults to port
8000 and proxies to `http://127.0.0.1:8010/api/qa`. Override with `CLARIVUE_PORT`
and `CLARIVUE_RAG_URL`. `RIS_KNOWLEDGE_GRAPH` enables graph enrichment in either
startup mode. Health endpoints report process liveness, not dependency readiness.

Regression checks:

```sh
.venv/bin/python -m pytest tests/http tests/qa tests/llm tests/retrieval/test_retrieval.py tests/retrieval/test_hybrid.py indexer/knowledge-infusion/tests -q
node --test tests/http/frontend.test.cjs
```

HTTP tests use controlled retrieval/model fixtures, including proxy errors,
invalid input, empty evidence, and citation mapping. Live checks additionally
require the actual database, embedding model, and Ollama services.

The hosted Worker also forwards `/api/qa` to `CLARIVUE_RAG_URL` (HTTPS), with
optional `CLARIVUE_RAG_TOKEN` for an authenticated upstream gateway. A deployed
Worker cannot reach this machine's localhost. This change does not publish the
site or expose the local research service publicly. Without a configured remote
endpoint the Worker returns an explicit 503, never a canned answer.
