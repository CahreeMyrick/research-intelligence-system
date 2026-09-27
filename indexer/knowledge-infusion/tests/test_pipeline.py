import importlib
import json
import networkx as nx
import pytest
from indexer.retrieval.service import EvidenceContext, EvidenceItem
from indexer.qa.service import QuestionAnsweringService
from indexer.qa.provider import MockLLMProvider

kg = importlib.import_module('indexer.knowledge-infusion')
ops = importlib.import_module('indexer.knowledge-infusion.operations')


def graph():
    g = nx.MultiDiGraph()
    for node, kind, aliases in [
        ('ChestPain', 'Symptom', ['chest pains']), ('Running', 'Activity', ['running']),
        ('Exercise', 'Activity', []), ('Exertion', 'Activity', []),
        ('CandidateA', 'Condition', []), ('CandidateB', 'Condition', [])
    ]:
        g.add_node(node, entity_type=kind, aliases=aliases)
    for source, relation, target in [
        ('Running', 'IS_A', 'Exercise'), ('Exercise', 'IS_A', 'Exertion'),
        ('CandidateA', 'HAS_SYMPTOM', 'ChestPain'), ('CandidateA', 'TRIGGERED_BY', 'Exertion'),
        ('CandidateB', 'HAS_SYMPTOM', 'ChestPain'),
    ]:
        g.add_edge(source, target, relation=relation)
    return g


def parsed(negated=False):
    return kg.ParsedQuery(intent='cause_explanation', entities=[
        {'mention': 'chest pains', 'type': 'Symptom', 'negated': negated},
        {'mention': 'running', 'type': 'Activity'}], relations=[
        {'source': 'chest pains', 'relation': 'OCCURS_DURING', 'target': 'running'}])


def context(g, query=None, finder=None):
    linked = ops.link_query(query or parsed(), g)
    return (finder or kg.PathFinder()).discover(ops.normalize_query(linked, g), g)


def test_constraint_intersection_and_direction():
    result = context(graph())
    assert [c.id for c in result.candidate_entities] == ['CandidateA']
    assert len(result.paths) == 1
    assert {e.relation for e in result.paths[0].edges} == {'HAS_SYMPTOM', 'TRIGGERED_BY'}
    assert len(result.normalized.ontology_edges) == 2
    assert result.normalized.expansions['Running'] == ['Exercise', 'Exertion', 'Running']
    assert all(e.quality == 0 for e in result.paths[0].edges)


def test_reversed_trigger_is_not_a_match():
    g = graph()
    g.remove_edge('CandidateA', 'Exertion')
    g.add_edge('Exertion', 'CandidateA', relation='TRIGGERED_BY')
    assert not context(g).paths


def test_ambiguous_and_negated_mentions_are_not_positive_seeds():
    g = graph()
    g.add_node('OtherPain', entity_type='Symptom', aliases=['chest pains'])
    linked = ops.link_query(parsed(), g)
    assert linked.entities[0].kg_id is None
    assert linked.relations == []
    result = context(graph(), parsed(negated=True))
    assert 'ChestPain' not in result.normalized.expansions


def test_budget_and_cycles():
    g = graph()
    g.add_edge('Exertion', 'Running', relation='IS_A')
    result = context(g, finder=kg.PathFinder(max_expansions=1))
    assert result.truncated
    assert len(result.normalized.expansions['Running']) == 3


def test_invalid_parser_output_fails_explicitly():
    with pytest.raises(ValueError):
        kg.LLMQueryParser(MockLLMProvider(response='not json')).parse('question')
    bad = parsed().model_dump()
    bad['relations'][0]['source'] = 'fabricated'
    with pytest.raises(ValueError):
        kg.ParsedQuery.model_validate(bad)


def test_parser_prunes_phantom_relations():
    raw_llm = json.dumps({
        'intent': 'cause',
        'entities': [{'mention': 'chest pains', 'type': 'symptom'}],
        'relations': [{'source': 'unknown', 'relation': 'CAUSES', 'target': 'chest pains'}],
        'attributes': []
    })
    res = kg.LLMQueryParser(MockLLMProvider(response=raw_llm)).parse('What causes chest pains?')
    assert len(res.entities) == 1
    assert res.relations == []


def test_case_insensitive_and_synonym_type_linking():
    g = graph()
    query = kg.ParsedQuery(intent='cause', entities=[
        {'mention': 'chest pains', 'type': 'symptom'},
        {'mention': 'running', 'type': 'activity'}
    ], relations=[{'source': 'chest pains', 'relation': 'OCCURS_DURING', 'target': 'running'}])
    linked = ops.link_query(query, g)
    assert linked.entities[0].kg_id == 'ChestPain'
    assert linked.entities[1].kg_id == 'Running'
    assert linked.relations[0].source == 'ChestPain'
    assert linked.relations[0].target == 'Running'



class Retrieval:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, **options):
        self.calls.append((question, options))
        return EvidenceContext(query=question, items=[EvidenceItem(
            evidence_id='evidence_1', document_id='doc', document_title='Test',
            publication_year=2020, section_title='Results', section_path=[],
            chunk_id='chunk', source_text='Supplied evidence.', page_start=None,
            page_end=None, relevance_score=.8)])


def test_fusion_provenance_filters_and_final_prompt():
    retrieval = Retrieval()
    parser = kg.LLMQueryParser(MockLLMProvider(response=parsed().model_dump_json()))
    adapter = kg.KnowledgeInfusedRetrievalService(retrieval, parser, graph())
    evidence = adapter.retrieve('chest pains while running', top_k=2, year_from=2019)
    assert len(evidence.items) == 1
    assert len(evidence.provenance['evidence_1']) == 2
    assert all(options['year_from'] == 2019 for _, options in retrieval.calls)
    assert evidence.provenance['evidence_1'][1].query.path_id == 'path_1'
    llm = MockLLMProvider(response='Supported [Evidence 1]')
    answer = QuestionAnsweringService(adapter, llm).answer('chest pains while running')
    assert answer.citations[0].chunk_id == 'chunk'
    assert 'not clinical evidence' in llm.prompts[0]
    assert 'OCCURS_DURING' in llm.prompts[0]


def test_unknown_query_still_retrieves_original():
    retrieval = Retrieval()
    parser = kg.LLMQueryParser(MockLLMProvider(response='{"intent":"definition"}'))
    result = kg.KnowledgeInfusedRetrievalService(retrieval, parser, graph()).retrieve('unknown')
    assert len(result.retrieval_queries) == 1
    assert not result.graph_context.paths
    assert result.items
