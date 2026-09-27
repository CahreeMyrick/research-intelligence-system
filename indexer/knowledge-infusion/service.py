"""Drop-in retrieval adapter; uses the existing embedding/search infrastructure."""
import json
from dataclasses import dataclass, field, replace
from .models import GraphContext, RetrievalQuerySpec, RetrievalTrace
from .operations import link_query, normalize_query, PathFinder, build_retrieval_queries
from indexer.retrieval.service import EvidenceContext
from indexer.retrieval.models import RetrievalQuery


@dataclass
class KnowledgeEvidenceContext(EvidenceContext):
    graph_context: GraphContext | None = None
    retrieval_queries: list[RetrievalQuerySpec] = field(default_factory=list)
    provenance: dict[str, list[RetrievalTrace]] = field(default_factory=dict)

    def to_prompt_context(self):
        graph = self.graph_context.model_dump_json() if self.graph_context else "{}"
        return (
            "QUERY STRUCTURE AND GRAPH ASSOCIATIONS (retrieval hints, not clinical evidence)\n"
            + graph + "\nDo not treat graph associations or scores as diagnoses or probabilities. "
            "Only retrieved passages support factual claims. Communicate uncertainty. "
            "Preserve negated findings; missing graph edges do not rule out conditions.\n\n"
            + super().to_prompt_context()
        )


class KnowledgeInfusedRetrievalService:
    def __init__(self, retrieval_service, parser, graph, path_finder=None):
        self.retrieval_service = retrieval_service
        self.parser = parser
        self.graph = graph
        self.path_finder = path_finder or PathFinder()

    def retrieve(self, question, **options):
        # Validate before invoking a model or repository.
        validated = RetrievalQuery(text=question, **options)
        parsed = self.parser.parse(validated.text)
        linked = link_query(parsed, self.graph)
        normalized = normalize_query(linked, self.graph)
        context = self.path_finder.discover(normalized, self.graph)
        queries = build_retrieval_queries(validated.text, context, self.graph)
        records, scores, provenance = {}, {}, {}
        for query in queries:
            result = self.retrieval_service.retrieve(query.text, **options)
            seen = set()
            for rank, item in enumerate(result.items, 1):
                key = (item.document_id, item.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                records.setdefault(key, item)
                # Reciprocal rank fusion avoids comparing differently calibrated search scores.
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                provenance.setdefault(key, []).append(RetrievalTrace(
                    query=query, rank=rank, retrieval_score=item.relevance_score))
        items, traces, counts = [], {}, {}
        for key in sorted(records, key=lambda k: (-scores[k], k)):
            item = records[key]
            if counts.get(item.document_id, 0) >= validated.max_results_per_document:
                continue
            counts[item.document_id] = counts.get(item.document_id, 0) + 1
            item = replace(item, evidence_id=f"evidence_{len(items)+1}", relevance_score=scores[key])
            items.append(item)
            traces[item.evidence_id] = provenance[key]
            if len(items) >= validated.top_k:
                break
        return KnowledgeEvidenceContext(query=validated.text, items=items, graph_context=context,
                                        retrieval_queries=queries, provenance=traces)


def load_graph(path):
    """Load an explicitly supplied graph; no demo medical assertions are loaded implicitly."""
    import networkx as nx
    from .models import Edge
    with open(path) as stream:
        payload = json.load(stream)
    graph = nx.MultiDiGraph()
    graph.graph.update(payload.get("metadata", {}))
    for node in payload["nodes"]:
        properties = dict(node)
        node_id = properties.pop("id")
        if node_id in graph or not isinstance(node_id, str) or not node_id:
            raise ValueError("Node IDs must be unique nonempty strings")
        if not isinstance(properties.get("entity_type"), str):
            raise ValueError("Every node requires an entity_type")
        graph.add_node(node_id, **properties)
    for raw in payload["edges"]:
        edge = Edge.model_validate(raw)
        if edge.source not in graph or edge.target not in graph:
            raise ValueError("Edge endpoints must exist")
        properties = {k: v for k, v in raw.items() if k not in {"source", "target"}}
        properties.update(edge.model_dump(exclude={"source", "target"}))
        graph.add_edge(edge.source, edge.target, **properties)
    return graph
