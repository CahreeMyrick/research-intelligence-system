# Knowledge-infused retrieval

All integration implementation and its tests live here. The existing
`knowledge_graph.py` is a visualization/demo, not the runtime knowledge source.
The CLI enables this adapter explicitly; ordinary retrieval remains available.

```sh
python -m indexer.qa.cli "Why have I been developing chest pains while running?" \
  --knowledge-graph /path/to/curated-graph.json
```

Install the project's `graph` extra to use the JSON graph loader. The parser uses
the configured answer LLM, in a separate call, and validates its JSON response.
Invalid parser output raises a validation error rather than silently inventing a
query. Other parsers may implement `parse(question) -> ParsedQuery`.

Because the requested directory contains a hyphen, Python callers use:

```python
from importlib import import_module
knowledge = import_module("indexer.knowledge-infusion")
retrieval = knowledge.KnowledgeInfusedRetrievalService(
    retrieval_service=existing_retrieval_service,
    parser=knowledge.LLMQueryParser(llm_provider),
    graph=knowledge.load_graph("curated-graph.json"),
)
# Pass retrieval to the existing QuestionAnsweringService, or inspect directly:
evidence = retrieval.retrieve("chest pains while running")
print(evidence.graph_context.model_dump_json(indent=2))
print(evidence.provenance)
```

## Stage contracts

| Stage | Input | Output |
| --- | --- | --- |
| Understanding/extraction | Question string | `ParsedQuery`: intent, typed mentions, negation, relations, attributes |
| Linking | Parsed query + graph alias vocabulary | `LinkedQuery`: canonical IDs or unresolved mentions; canonical positive relations |
| Normalization | Linked query + ontology | `NormalizedQuery`: original concepts, ancestor sets, supporting directed ontology edges |
| Traversal/ranking | Normalized query + graph | `GraphContext`: ranked paths, candidates, search truncation flag |
| Query construction | Original question + graph context | `RetrievalQuerySpec[]`: original query plus one query per retained path |
| Embedding/retrieval | Each retrieval query + unchanged filters | Existing retrieval service's `EvidenceContext` |
| Fusion | Evidence from all queries | `KnowledgeEvidenceContext`: unique chunks, reciprocal-rank scores, all retrieval traces |
| Generation | Fused context + question | Existing QA service's cited `AnswerResponse` |

Graph node JSON requires `id`, `entity_type`, with optional `name`, `aliases`, and
`external_ids`. Edge JSON requires `source`, `relation`, `target`, with optional
`evidence_source` and `quality` (0–1). Top-level keys are `nodes` and `edges`.
External medical IDs are supplied, never guessed. For example, a *synthetic*
(nonmedical) graph file is:

```json
{
  "nodes": [
    {"id": "FindingA", "entity_type": "Symptom", "aliases": ["finding a"]},
    {"id": "ConditionA", "entity_type": "Condition"}
  ],
  "edges": [
    {"source": "ConditionA", "relation": "HAS_SYMPTOM", "target": "FindingA"}
  ]
}
```

## Traversal and interpretation

Alias linking is case-insensitive, exact, and type-constrained. A unique match
has confidence 1 as a deterministic alias match, not a clinical probability.
Ambiguous/unknown mentions remain unresolved. Configure aliases such as
`chest pains` explicitly. Negated mentions are preserved but excluded as positive
seeds. Recurrence is not inferred simply from "have been developing".

Normalization follows outgoing IS_A, SUBCLASS_OF, SYNONYM_OF edges for at most two
hops and 100 concepts per mention. PART_OF does not imply interchangeable concepts.
Clinical traversal preserves edge direction while exploring either endpoint,
allows only approved clinical relations, and excludes repeated nodes. Defaults:
three hops, 2,000 edge examinations, eight returned paths. OCCURS_DURING requires
an appropriate condition-to-trigger edge and the symptom concept in the path.
Multiple positive concepts favor connecting paths; single concepts support
condition expansion. Strict intersection can miss relevant alternatives; Q0 is
always retained so evidence search is not confined to graph candidates.

Ranking combines query-concept coverage (0.35), relation relevance (0.30), inverse
path length (0.20), and supplied source quality (0.15). Unsourced edges contribute
zero source quality. Scores are heuristics, not diagnostic probabilities. Fusion
uses reciprocal rank with constant 60, deduplicates document/chunk pairs, retains
all retrieval routes, reapplies document diversity and top-k limits, and renumbers
citations. Provenance is available on the retrieval result; the existing final
AnswerResponse schema is unchanged.

## Current scope

This is a text-query integration foundation. It does not interpret uploaded
images, verify source quality, provide medical triage, infer temporal/negative
clinical constraints, or implement full ontology reasoning. Attributes and intent
are retained for generation; only OCCURS_DURING currently imposes a specific
relation constraint. Other relations use the bounded clinical relation allowlist.
Model extraction and curated graph quality need domain evaluation before clinical
use. No sample associations are automatically loaded into production retrieval.

Useful evaluation prompts include a plain-language definition, a professional
comparison, a negated finding, a symptom with a trigger, an unfamiliar concept,
and an evidence/provenance request. Test unknown and ambiguous links explicitly.

```sh
.venv/bin/python -m pytest indexer/knowledge-infusion/tests tests/qa -q
```

## Run with the existing demo graph

The included `demo_graph.json` is exported from `knowledge_graph.py`. It preserves
canonical concepts, synthetic image observations, node attributes, and relations.
Its associations are unverified demo data with quality zero, not clinical evidence.
Regenerate it after editing the Python demo:

```sh
.venv/bin/python -m indexer.knowledge-infusion.export_graph
.venv/bin/python -m indexer.qa.cli \
  "What conditions are associated with pleural effusion and cardiomegaly?" \
  --knowledge-graph indexer/knowledge-infusion/demo_graph.json
```

This graph contains chest X-ray concepts; it does not yet contain ChestPain,
Running, PhysicalExertion, or Angina. The running/chest-pain example therefore
falls back to original-query retrieval. Adding those concepts requires explicit
nodes, aliases, ontology edges and sourced clinical relationships. The exporter
does not extract or verify knowledge from your indexed papers.
