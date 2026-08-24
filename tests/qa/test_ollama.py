from __future__ import annotations

import json

from indexer.qa.ollama import OllamaQwenProvider


def test_ollama_provider_properties():
    provider = OllamaQwenProvider()

    assert provider.name == "ollama:qwen3:4b-docmind"
    assert provider.version == "qwen3:4b-docmind"


def test_ollama_provider_generate(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "response": "LOCAL QWEN WORKS",
                }
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "indexer.qa.ollama.urlopen",
        fake_urlopen,
    )

    provider = OllamaQwenProvider()

    result = provider.generate("Say hello.")

    assert result == "LOCAL QWEN WORKS"
    assert captured["timeout"] == 120.0

    payload = json.loads(
        captured["request"].data.decode("utf-8")
    )

    assert payload["model"] == "qwen3:4b-docmind"
    assert payload["prompt"] == "Say hello."
    assert payload["stream"] is False
