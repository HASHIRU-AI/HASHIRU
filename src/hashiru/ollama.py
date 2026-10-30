"""Ollama's HTTP API: load and unload local models and read their sizes."""
from typing import Any

import httpx

from hashiru.llm import is_local

__all__ = ["Ollama", "tag"]


def tag(model: str) -> str:
    """LiteLLM string -> Ollama tag: ollama_chat/qwen3.5:4b -> qwen3.5:4b."""
    return model.split("/", 1)[1]


class Ollama:
    def __init__(self, url: str, models: tuple[str, ...], num_ctx: int = 16384) -> None:
        self.url = url
        self.local = {tag(m) for m in models if is_local(m)}
        self.num_ctx = num_ctx  # load with the context window agents' calls use, or Ollama reloads the model

    async def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET (no body) or POST to the Ollama HTTP API."""
        async with httpx.AsyncClient(base_url=self.url, timeout=600) as http:
            resp = await (http.get(path) if body is None else http.post(path, json=body))
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result

    async def load(self, model: str) -> int:
        """Load a local model now so its memory is really in use. Returns the MB it takes loaded (weights plus its
        context window), or its file size if Ollama does not report it."""
        await self.request("/api/generate", {"model": tag(model), "keep_alive": "30m", "stream": False,
                                             "options": {"num_ctx": self.num_ctx}})
        running = (await self.request("/api/ps")).get("models", [])
        loaded = next((m["size"] for m in running if m.get("name") == tag(model) and m.get("size")), 0)
        return loaded // 2**20 if loaded else (await self.sizes()).get(tag(model), 0)

    async def unload(self, model: str) -> None:
        await self.request("/api/generate", {"model": tag(model), "keep_alive": 0, "stream": False})

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Every pulled model by tag: its capabilities, details (parameter_size), context length and size."""
        out = {}
        for t in (await self.request("/api/tags"))["models"]:
            shown = await self.request("/api/show", {"model": t["name"]})
            info = {k: v for k, v in (shown.get("model_info") or {}).items() if k.endswith(".context_length")}
            out[t["name"]] = {"capabilities": shown.get("capabilities"), "details": shown.get("details") or {},
                              "model_info": info, "size": t.get("size")}
        return out

    async def sizes(self) -> dict[str, int]:
        """Size in MB of each configured local model, from /api/tags."""
        if not self.local:
            return {}
        tags = (await self.request("/api/tags"))["models"]
        return {t["name"]: t["size"] // 2**20 for t in tags if t["name"] in self.local}
