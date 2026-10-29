"""Ollama's HTTP API: load and unload local models and read their sizes."""
from typing import Any

import httpx

from hashiru.llm import is_local

__all__ = ["Ollama", "tag"]


def tag(model: str) -> str:
    """LiteLLM string -> Ollama tag: ollama_chat/qwen3.5:4b -> qwen3.5:4b."""
    return model.split("/", 1)[1]


class Ollama:
    def __init__(self, url: str, models: tuple[str, ...]) -> None:
        self.url = url
        self.local = {tag(m) for m in models if is_local(m)}

    async def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET (no body) or POST to the Ollama HTTP API."""
        async with httpx.AsyncClient(base_url=self.url, timeout=600) as http:
            resp = await (http.get(path) if body is None else http.post(path, json=body))
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result

    async def load(self, model: str) -> int:
        """Load a local model now so its memory is really in use. Returns its size in MB."""
        await self.request("/api/generate", {"model": tag(model), "keep_alive": "30m", "stream": False})
        return (await self.sizes()).get(tag(model), 0)

    async def unload(self, model: str) -> None:
        await self.request("/api/generate", {"model": tag(model), "keep_alive": 0, "stream": False})

    async def sizes(self) -> dict[str, int]:
        """Size in MB of each configured local model, from /api/tags."""
        if not self.local:
            return {}
        tags = (await self.request("/api/tags"))["models"]
        return {t["name"]: t["size"] // 2**20 for t in tags if t["name"] in self.local}
