from __future__ import annotations

import logging
from typing import Iterable

from indexer.retrieval.models import CandidateChunk, RetrievalQuery

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines semantic and lexical retrieval candidates.

    Responsibilities:
    - merge candidates by chunk_id
    - normalize scores from each retrieval method
    - apply semantic/lexical weights
    - enforce max_results_per_document
    - return candidates ordered by final score
    """

    def combine(
        self,
        query: RetrievalQuery,
        semantic_candidates: Iterable[CandidateChunk],
        lexical_candidates: Iterable[CandidateChunk],
    ) -> list[CandidateChunk]:
        semantic = list(semantic_candidates)
        lexical = list(lexical_candidates)

        candidates: dict[str, CandidateChunk] = {}

        # Merge semantic candidates.
        for candidate in semantic:
            candidates[candidate.chunk_id] = CandidateChunk(
                **candidate.model_dump()
            )

        # Merge lexical candidates.
        for candidate in lexical:
            existing = candidates.get(candidate.chunk_id)

            if existing is None:
                candidates[candidate.chunk_id] = CandidateChunk(
                    **candidate.model_dump()
                )
                continue

            if candidate.lexical_score is not None:
                existing.lexical_score = candidate.lexical_score

            existing.matched_methods.update(candidate.matched_methods)

        # Normalize scores independently.
        semantic_scores = [
            c.semantic_score
            for c in candidates.values()
            if c.semantic_score is not None
        ]

        lexical_scores = [
            c.lexical_score
            for c in candidates.values()
            if c.lexical_score is not None
        ]

        semantic_normalized = self._normalize_scores(semantic_scores)
        lexical_normalized = self._normalize_scores(lexical_scores)

        semantic_index = 0
        lexical_index = 0

        for candidate in candidates.values():
            semantic_score = 0.0
            lexical_score = 0.0

            if candidate.semantic_score is not None:
                semantic_score = semantic_normalized[semantic_index]
                semantic_index += 1

            if candidate.lexical_score is not None:
                lexical_score = lexical_normalized[lexical_index]
                lexical_index += 1

            candidate.final_score = (
                query.semantic_weight * semantic_score
                + query.lexical_weight * lexical_score
            )

        ranked = sorted(
            candidates.values(),
            key=lambda c: c.final_score,
            reverse=True,
        )

        # Enforce document diversity.
        results: list[CandidateChunk] = []
        document_counts: dict[str, int] = {}

        for candidate in ranked:
            count = document_counts.get(candidate.document_id, 0)

            if count >= query.max_results_per_document:
                continue

            document_counts[candidate.document_id] = count + 1
            results.append(candidate)

            if len(results) >= query.top_k:
                break

        logger.debug(
            "Hybrid fusion produced %d results from %d unique candidates",
            len(results),
            len(candidates),
        )

        return results

    @staticmethod
    def _normalize_scores(scores: list[float | None]) -> list[float]:
        """
        Min-max normalize scores to [0, 1].

        If all scores are identical, return 1.0 for each score rather than
        collapsing them to zero. Presence of a retrieval match should still
        contribute to the fused ranking.
        """
        values = [float(score) for score in scores if score is not None]

        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return [1.0 for _ in values]

        return [
            (value - minimum) / (maximum - minimum)
            for value in values
        ]
