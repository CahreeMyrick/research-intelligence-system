from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChunkerConfig:
    """Explicit, versioned configuration for structure-aware semantic chunking."""
    target_tokens: int = 512
    overlap_tokens: int = 64
    max_tokens: int = 512
    chunker_version: str = "1.0"

    def config_hash(self) -> str:
        """Deterministic SHA-256 hash of configuration parameters."""
        payload = {
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "max_tokens": self.max_tokens,
            "chunker_version": self.chunker_version,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]
