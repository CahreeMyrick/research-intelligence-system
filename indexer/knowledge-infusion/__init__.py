"""Import with importlib.import_module('indexer.knowledge-infusion').

Absolute dynamic imports also allow tools to inspect this hyphenated directory.
"""
from importlib import import_module

_models = import_module('indexer.knowledge-infusion.models')
_operations = import_module('indexer.knowledge-infusion.operations')
_service = import_module('indexer.knowledge-infusion.service')
ParsedQuery = _models.ParsedQuery
LinkedQuery = _models.LinkedQuery
GraphContext = _models.GraphContext
GraphPath = _models.GraphPath
LLMQueryParser = _operations.LLMQueryParser
PathFinder = _operations.PathFinder
KnowledgeInfusedRetrievalService = _service.KnowledgeInfusedRetrievalService
KnowledgeEvidenceContext = _service.KnowledgeEvidenceContext
load_graph = _service.load_graph

__all__ = ['ParsedQuery', 'LinkedQuery', 'GraphContext', 'GraphPath', 'LLMQueryParser',
           'PathFinder', 'KnowledgeInfusedRetrievalService', 'KnowledgeEvidenceContext', 'load_graph']
