from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaQwenProvider:
    """
    Local Qwen provider backed by Ollama.

    Keeps Ollama-specific HTTP behavior isolated from the QA layer.
    """

    def __init__(
        self,
        model: str = "qwen3:4b-docmind",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def version(self) -> str:
        return self.model

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}: {body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at "
                f"{self.base_url}: {exc}"
            ) from exc

        answer = data.get("response")

        if not isinstance(answer, str):
            raise RuntimeError(
                f"Unexpected Ollama response: {data!r}"
            )

        return answer.strip()
