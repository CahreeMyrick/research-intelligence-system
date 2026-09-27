"""Serializable contracts for every knowledge-infusion boundary."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class EntityMention(BaseModel):
    mention: str
    type: str
    negated: bool = False


class RelationMention(BaseModel):
    source: str
    relation: str
    target: str


class Attribute(BaseModel):
    entity: str
    attribute: str
    value: str


class ParsedQuery(BaseModel):
    intent: str
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self):
        mentions = {e.mention for e in self.entities}
        if any(r.source not in mentions or r.target not in mentions for r in self.relations):
            raise ValueError("Relations must reference extracted mentions")
        if any(a.entity not in mentions for a in self.attributes):
            raise ValueError("Attributes must reference extracted mentions")
        return self


class LinkedEntity(EntityMention):
    kg_id: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    external_ids: dict[str, str] = Field(default_factory=dict)


class LinkedQuery(BaseModel):
    parsed: ParsedQuery
    entities: list[LinkedEntity]
    relations: list[RelationMention]


class Edge(BaseModel):
    source: str
    relation: str
    target: str
    evidence_source: str | None = None
    quality: float = Field(default=0, ge=0, le=1)


class NormalizedQuery(BaseModel):
    linked: LinkedQuery
    # Preserve the original concept and each justified ancestor, not a replacement.
    expansions: dict[str, list[str]]
    ontology_edges: list[Edge]


class GraphPath(BaseModel):
    id: str
    nodes: list[str]
    edges: list[Edge]  # Stored direction is preserved even when traversed backwards.
    score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_path(self):
        if len(self.nodes) != len(self.edges) + 1:
            raise ValueError("Path must have one more node than edges")
        for a, b, edge in zip(self.nodes, self.nodes[1:], self.edges):
            if {a, b} != {edge.source, edge.target}:
                raise ValueError("Path edge does not connect consecutive nodes")
        return self


class CandidateEntity(BaseModel):
    id: str
    type: str
    score: float
    path_ids: list[str]


class GraphContext(BaseModel):
    normalized: NormalizedQuery
    candidate_entities: list[CandidateEntity]
    paths: list[GraphPath]
    truncated: bool = False


class RetrievalQuerySpec(BaseModel):
    id: str
    text: str
    source: Literal["original", "graph_path"]
    path_id: str | None = None


class RetrievalTrace(BaseModel):
    query: RetrievalQuerySpec
    rank: int
    retrieval_score: float
