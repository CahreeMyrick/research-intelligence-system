from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizers used in chunking and embedding."""
    def count_tokens(self, text: str) -> int:
        ...

    def tokenize(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...


class HuggingFaceTokenizer:
    """Tokenizer backed by HuggingFace / Transformers AutoTokenizer."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class RegexFallbackTokenizer:
    """
    Lightweight deterministic tokenizer for testing or environments
    where full transformer tokenizer is not loaded.
    """
    # Matches words, numbers, or standalone punctuation symbols
    TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.TOKEN_RE.findall(text))

    def tokenize(self, text: str) -> list[int]:
        # Hash each token to an integer token id
        tokens = self.TOKEN_RE.findall(text)
        return [abs(hash(t)) % 100000 for t in tokens]

    def decode(self, tokens: list[int]) -> str:
        return ""  # Used primarily for token counts in fallback
