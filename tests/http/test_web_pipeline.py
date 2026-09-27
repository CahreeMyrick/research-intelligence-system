"""HTTP contract coverage through the web proxy and real answer service."""
import importlib.util
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from indexer import http_api
from indexer.qa.service import QuestionAnsweringService
from indexer.retrieval.service import EvidenceContext, EvidenceItem


@contextmanager
def server(handler):
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}'
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.fixture
def pipeline(monkeypatch):
    class Retrieval:
        questions = []
        items = [EvidenceItem('evidence_1', 'doc', 'Research title', 2024,
                              'Results', ['Results'], 'chunk', 'Verbatim evidence.', 1, 1, .9)]
        def retrieve(self, question, **options):
            self.questions.append(question)
            return EvidenceContext(question, self.items)
    class LLM:
        prompts = []
        def generate(self, prompt):
            self.prompts.append(prompt)
            return 'Supported finding [Evidence 1]. Unsupported reference [Evidence 99].'
    retrieval, llm = Retrieval(), LLM()
    monkeypatch.setattr(http_api, 'SERVICE', retrieval)
    monkeypatch.setattr(http_api, 'CONFIG', SimpleNamespace(top_k=10, candidate_k=50, embedding_model='test', embedding_dimensions=768, embedding_model_version='1.0', llm_model='test'))
    monkeypatch.setattr(http_api, 'QA', QuestionAnsweringService(retrieval, llm))
    return retrieval, llm


def post(url, payload):
    request = Request(url + '/api/qa', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def test_proxy_to_retrieval_generation_and_citations(pipeline, monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'clarivue_qa/api_server.py'
    if not path.exists():
        pytest.skip('Sibling Clarivue UI not checked out')
    spec = importlib.util.spec_from_file_location('web_proxy', path)
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    with server(http_api.Handler) as backend:
        monkeypatch.setattr(proxy, 'RESEARCH_RAG_URL', backend + '/api/qa')
        with server(proxy.Handler) as web:
            status, body = post(web, {'question': 'An example question'})
    assert status == 200
    assert pipeline[0].questions == ['An example question']
    assert 'Verbatim evidence.' in pipeline[1].prompts[0]
    assert body['sources'][0]['passage'] == 'Verbatim evidence.'
    assert body['citations'][0]['evidence_id'] == body['sources'][0]['id']
    assert len(body['citations']) == 1
    assert body['retrieval']['retrieved_count'] == 1
    assert body['answer'] == '\n\n'.join(body['paragraphs'])
    # A dead backend returns JSON, never a dropped socket or a demo answer.
    with server(proxy.Handler) as web:
        status, body = post(web, {'question': 'Another question'})
    assert status == 502
    assert body['error'] == 'backend_unavailable'


@pytest.mark.parametrize('payload', [{}, [], {'question': None}, {'question': ' '}, {'question': 'x' * 1501}])
def test_invalid_queries_do_not_reach_retrieval(pipeline, payload):
    with server(http_api.Handler) as backend:
        status, body = post(backend, payload)
    assert status == 400
    assert body['error'] == 'invalid_request'
    assert pipeline[0].questions == []


def test_no_evidence_skips_generation(pipeline):
    pipeline[0].items = []
    body = http_api.evidence_response('Unknown subject')
    assert body['sources'] == body['citations'] == []
    assert 'insufficient evidence' in body['answer']
    assert pipeline[1].prompts == []


def test_service_failure_returns_json(pipeline, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('database unavailable')
    monkeypatch.setattr(pipeline[0], 'retrieve', fail)
    with server(http_api.Handler) as backend:
        status, body = post(backend, {'question': 'test'})
    assert status == 502
    assert body['message'] == 'database unavailable'


def test_graph_context_survives_http_response(pipeline, monkeypatch):
    from importlib import import_module
    import networkx as nx
    knowledge = import_module('indexer.knowledge-infusion')
    graph = nx.MultiDiGraph(status='demo_unverified')
    graph.add_node('Finding', entity_type='Symptom', aliases=['finding'])
    graph.add_node('Condition', entity_type='Condition')
    graph.add_edge('Condition', 'Finding', relation='HAS_SYMPTOM', quality=0)
    class Parser:
        def parse(self, question):
            return knowledge.ParsedQuery(intent='question', entities=[{'mention':'finding','type':'Symptom'}])
    retrieval = knowledge.KnowledgeInfusedRetrievalService(pipeline[0], Parser(), graph)
    monkeypatch.setattr(http_api, 'SERVICE', retrieval)
    monkeypatch.setattr(http_api, 'QA', QuestionAnsweringService(retrieval, pipeline[1]))
    body = http_api.evidence_response('What causes finding?')
    assert body['entities'][0]['kg_id'] == 'Finding'
    assert body['graph_paths'][0]['edges'][0]['relation'] == 'HAS_SYMPTOM'
    assert body['knowledge_graph']['metadata']['status'] == 'demo_unverified'
    assert body['retrieval']['mode'] == 'knowledge_infused_hybrid'
    assert 'QUERY STRUCTURE AND GRAPH ASSOCIATIONS' in pipeline[1].prompts[0]


def test_web_default_selects_medical_database_and_vectors(monkeypatch):
    monkeypatch.setattr(http_api, 'CONFIG_PATH', http_api.DEFAULT_CONFIG_PATH)
    service, config = http_api.build_retrieval_service()
    assert http_api.DEFAULT_CONFIG_PATH.name == 'clinivue_config.yaml'
    assert config.database_url == 'postgresql://clinivue@localhost:5433/clinivue'
    assert service.repository.db.database_url == config.database_url
    assert service.repository.embedding_provider.name == config.embedding_model
    assert service.repository.embedding_provider.dimensions == config.embedding_dimensions == 768
    assert service.repository.embedding_provider.version == config.embedding_model_version
