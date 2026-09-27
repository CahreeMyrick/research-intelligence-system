# paper-ingest

A simple ingestion pipeline implementing:

```
ingest: P(PDF) -> Corpus
```

Turns a folder of research-paper PDFs into a canonical, structured `Corpus`
(title/authors/abstract, a section/subsection tree with paragraphs, and a
parsed reference list). This is **ingestion only** — no retrieval,
embeddings, RAG, or LLM reasoning, and (per current scope) no
tables/figures extraction.

## Install

```bash
pip install -r requirements.txt
# or, as an editable package:
pip install -e .
```

Dependencies: `pymupdf` (layout-aware text extraction), `pydantic`
(canonical schema + JSON export). See the top of `requirements.txt` for why
each is there.

## Usage

```bash
python -m ingest.cli --papers_dir papers/ --out corpus -v
```

For PDFs nested in subdirectories, add `--recursive`:

```bash
python -m ingest.cli --papers_dir chest_xray_rag --recursive --out medical_corpus -v
```

The Python API supports the same option with `ingest("chest_xray_rag", output_dir="medical_corpus", recursive=True)`.
Discovery is non-recursive by default; non-PDF files are ignored.

## Clinivue question answering

With the Clinivue database and Ollama running, ask questions from this directory:

```bash
../.venv/bin/python -m indexer.qa.cli --config clinivue_config.yaml \
  "Which approaches to automated chest X-ray report generation are described in this corpus?"
```

The configuration selects the database, embedding model, answer model, and retrieval
limits. Use `--model`, `--top-k`, or `--candidate-k` to override those answer and
retrieval settings for a single question. Answers include retrieved evidence citations.

```python
from ingest import ingest

# Ingest and save to target directory structure
corpus = ingest("papers/", output_dir="corpus")
for doc in corpus.documents:
    print(doc.metadata.title, "->", len(doc.sections), "sections")
```

The output directory follows the canonical corpus structure:

```
corpus/
├── raw/
│   └── pdf/
│       ├── doc_001.pdf
│       └── ...
├── parsed/
│   ├── doc_001.json
│   └── ...
├── chunks/
├── embeddings/
└── manifest.json
```


Generate a synthetic test paper if you don't have one handy:

```bash
python scripts/make_sample_pdf.py papers/sample_001.pdf
```

## Pipeline stages

```
PDF
 │
 ▼
1. identity.py     -- sha256(bytes) -> stable id, independent of filename/path
 ▼
2. extraction.py   -- PyMuPDF: flat, ordered TextBlocks w/ font size + bold + page
 ▼
3. structure.py    -- heading detection (numbering / known names / font-size+bold)
 │                     -> Section tree, each Section holding its Paragraphs
 ▼
4. metadata.py     -- title/authors (page-1 layout heuristics), abstract (from
 │                     the tree), DOI / arXiv id / year (regex)
 ▼
5. references.py   -- locate References/Bibliography section, split into
 │                     entries, parse id/year/DOI, always keep raw_text
 ▼
6. schema.py + pipeline.py -- assemble into a validated Document -> Corpus
```

`pipeline.ingest()` isolates failures per-file (a corrupt PDF is logged and
skipped, not fatal to the whole run) and deduplicates by content hash.

## Known limitations (v1, by design)

- **Heuristic, not ML-based.** Heading/title/author detection uses font
  size, boldness, and regex patterns — no layout model, no GROBID. Works
  well on typical single-column academic PDFs (tested against an
  ICLR/NeurIPS-style layout); multi-column layouts, unusual templates and
  scanned/image-only PDFs are not yet handled.
- **`venue`, `affiliations`, and per-reference `authors`/`title` are left
  `null`/empty** rather than guessed — per the ingestion contract, "unknown"
  beats "invented." These are natural targets for a later enrichment step
  (e.g. resolving DOIs against Crossref) that's deliberately kept out of
  ingestion so it stays offline and fast.
- **No tables/figures extraction** — explicitly out of scope for this pass.
- **No paragraph-level bbox/position provenance** — sections/paragraphs
  carry page numbers, not (x, y) coordinates. Easy to add later since
  `extraction.py` already has access to per-span bounding boxes.

## Layout

```
paper_ingest/
├── pyproject.toml
├── requirements.txt
├── ingest/
│   ├── schema.py       # Corpus / Document / Section / Paragraph / Reference (pydantic)
│   ├── identity.py      # stage 1
│   ├── extraction.py    # stage 2
│   ├── structure.py     # stage 3
│   ├── metadata.py      # stage 4
│   ├── references.py    # stage 5
│   ├── pipeline.py      # orchestration: ingest(papers_dir) -> Corpus
│   └── cli.py
├── scripts/make_sample_pdf.py
└── tests/test_pipeline.py
```
