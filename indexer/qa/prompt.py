from __future__ import annotations

from indexer.retrieval.service import EvidenceContext


SYSTEM_INSTRUCTIONS = """You are a research assistant.

Answer the user's question using only the supplied evidence.

Rules:
1. Do not invent facts that are not supported by the evidence.
2. Do not rely on outside knowledge.
3. If the evidence is insufficient, explicitly say that the available
   evidence is insufficient.
4. Preserve important qualifications and uncertainty from the evidence.
5. Cite factual claims using the supplied evidence IDs.
6. Do not fabricate evidence IDs or citations.
"""


def build_answer_prompt(
    question: str,
    evidence: EvidenceContext,
) -> str:
    """
    Build a deterministic grounded-answer prompt.
    """

    return "\n".join(
        [
            SYSTEM_INSTRUCTIONS,
            "",
            "RETRIEVED EVIDENCE",
            "==================",
            evidence.to_prompt_context(),
            "",
            "USER QUESTION",
            "=============",
            question,
            "",
            "INSTRUCTIONS",
            "============",
            "Provide a concise research answer grounded exclusively in the "
            "retrieved evidence.",
            "Reference supporting evidence using [Evidence N].",
        ]
    )
