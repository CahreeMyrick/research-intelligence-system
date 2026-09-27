from pydantic import BaseModel

class StructuredQuery(BaseModel):
    intent: QueryIntent
    entities: list[QueryEntities]
    realtions: list[QueryEntityRelations]
    attributes: list[QueryAttributes]

class LinkedQuery(BaseModel):
    entities: list[dict]
    relations: lsit[dict]

class GraphPath(BaseModel):
    nodes: list[str]
    relations: list[str]
    score: float

class GraphContext(BaseModel):
    query_entities: list[LinkedEntity]
    candidate_entities: list[CandidateEntity]
    paths: list[GraphPath]


