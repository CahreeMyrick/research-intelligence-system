import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from evals.framework import (Benchmark, Question, RunSettings,
                            run_benchmark, validate_corpus)
from indexer.qa.provider import MockLLMProvider
from indexer.retrieval.service import EvidenceContext, EvidenceItem


@pytest.fixture
def evidence():
    return EvidenceContext(query="What?", items=[EvidenceItem(
        evidence_id="evidence_1", document_id="doc", document_title="Title",
        publication_year=None, section_title="Intro", section_path=[], chunk_id="chunk",
        source_text="Supported fact.", page_start=1, page_end=1, relevance_score=1)])


@pytest.fixture
def benchmark():
    return Benchmark(name="test", version="1", questions=[Question(
        id="q1", question="What?", expected_answer="Supported fact.",
        supporting_passages=[dict(document_id="doc", chunk_id="chunk", quote="Supported fact.")])])


class Judge:
    def grade(self, *args):
        return {name: {"passed": True} for name in
                ("faithfulness", "answer_relevancy", "citation_support")}


class Retrieval:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return self.evidence


def run(benchmark, evidence, text="Supported fact. [Evidence 1]", judge=None):
    retrieval = Retrieval(evidence)
    result = run_benchmark(benchmark, retrieval, MockLLMProvider(text), judge or Judge(), RunSettings())
    assert retrieval.calls == len(benchmark.questions)
    return result


def test_success(benchmark, evidence):
    result = run(benchmark, evidence)
    assert result['summary']['kr2']['rate'] == 1
    assert result['summary']['kr3']['rate'] == 1


@pytest.mark.parametrize('text', ['Supported fact.', 'Supported fact. [Evidence 1] [Evidence 99]', ''])
def test_bad_citations_fail_even_when_judge_passes(benchmark, evidence, text):
    result = run(benchmark, evidence, text)
    assert result['questions'][0]['kr2_pass']
    assert not result['questions'][0]['kr3_pass']


def test_generation_error_preserves_retrieval_hit(benchmark, evidence):
    class BrokenProvider:
        def generate(self, prompt):
            raise RuntimeError('offline')
    result = run_benchmark(benchmark, Retrieval(evidence), BrokenProvider(), Judge(), RunSettings())
    assert result['summary']['kr2']['rate'] == 1
    assert result['summary']['kr3']['rate'] == 0
    assert result['summary']['kr3']['total'] == 1


def test_judge_error_counts_as_failure(benchmark, evidence):
    class BrokenJudge:
        def grade(self, *args):
            raise RuntimeError('judge offline')
    result = run(benchmark, evidence, judge=BrokenJudge())
    assert result['summary']['kr3']['rate'] == 0
    assert 'Judge:' in result['questions'][0]['errors'][0]


def test_wrong_document_is_not_a_hit(benchmark, evidence):
    evidence.items[0] = replace(evidence.items[0], document_id='other')
    assert run(benchmark, evidence)['summary']['kr2']['rate'] == 0


def test_ninety_percent_boundary(benchmark, evidence):
    benchmark.questions = [benchmark.questions[0].model_copy(update={'id': str(i)}) for i in range(10)]
    class OneFailure(Judge):
        def grade(self, question, *args):
            metrics = super().grade()
            metrics['citation_support']['passed'] = question.id != '0'
            return metrics
    result = run(benchmark, evidence, judge=OneFailure())
    assert result['summary']['kr3'] == dict(passed=9, total=10, rate=0.9, status='met')


def test_reject_empty_duplicate_and_whitespace_benchmarks(benchmark):
    for update in ({'questions': []}, {'questions': benchmark.questions * 2}, {'name': ' '}):
        with pytest.raises(ValidationError):
            Benchmark.model_validate(benchmark.model_dump() | update)


def test_corpus_validation_catches_stale_labels(benchmark, tmp_path):
    path = tmp_path / 'doc.json'
    path.write_text(json.dumps({'chunks': [dict(document_id='doc', chunk_id='chunk', source_text='Supported fact.')]}))
    assert len(validate_corpus(benchmark, tmp_path)) == 64
    path.write_text(json.dumps({'chunks': []}))
    with pytest.raises(ValueError, match='missing or stale'):
        validate_corpus(benchmark, tmp_path)


def test_retrieval_failure_counts_in_both_denominators(benchmark):
    class BrokenRetrieval:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError('database offline')
    result = run_benchmark(benchmark, BrokenRetrieval(), MockLLMProvider(), Judge(), RunSettings())
    assert all(r['total'] == 1 and r['rate'] == 0 for r in result['summary'].values())


def test_metric_error_retains_other_scores(benchmark, evidence):
    class PartialJudge(Judge):
        def grade(self, *args):
            metrics = super().grade()
            metrics['citation_support'] = dict(passed=False, error='timeout')
            return metrics
    row = run(benchmark, evidence, judge=PartialJudge())['questions'][0]
    assert row['metrics']['faithfulness']['passed']
    assert not row['kr3_pass']
    assert row['errors'] == ['Judge citation_support: timeout']


def test_validate_cli(benchmark, tmp_path, capsys):
    from evals.run import main
    dataset = tmp_path / 'benchmark.json'
    dataset.write_text(benchmark.model_dump_json())
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'doc.json').write_text(json.dumps({'chunks': [dict(
        document_id='doc', chunk_id='chunk', source_text='Supported fact.')]}))
    assert main(['--dataset', str(dataset), '--corpus', str(corpus), '--validate-only']) == 0
    assert 'Valid: 1 questions' in capsys.readouterr().out
