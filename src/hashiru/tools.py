"""The built-in MCP tools. Each one learns who is calling from the request's bearer token, never from
an argument, so an agent cannot act as another agent."""
import asyncio
import ipaddress
import json
import math
import re
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from mcp.server.mcpserver import Context, MCPServer

from hashiru.ledger import AgentRecord, free_mb
from hashiru.llm import is_local
from hashiru.session import USER, Session

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["ROLES", "Tools", "load_memory", "non_public"]

ROLES = ("manager", "worker")
NAME = re.compile(r"[a-z0-9_]{1,32}")
RESERVED = {USER, "ceo"}  # an agent named "user" would get the human's token and speak to the CEO as them
MAX_REDIRECTS = 5
MAX_PAGE_BYTES = 2_000_000


def load_memory(data_dir: Path) -> dict[str, str]:
    """Each session starts from data/memory.json; 'add' changes only that session's copy."""
    path = data_dir / "memory.json"
    return dict(json.loads(path.read_text())) if path.exists() else {}


async def non_public(url: str) -> str | None:
    """Why `url` must not be fetched, or None. Agents read untrusted pages that can name any address, so
    loopback (Ollama, the hub), private networks and link-local (cloud metadata) are refused.

    The name is resolved here and again by the HTTP client, so DNS rebinding between the two is not covered.
    """
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host:
        return f"refusing {url}: only http(s) URLs with a host"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return f"could not resolve {parsed.host}: {exc}"
    for info in infos:
        if not ipaddress.ip_address(str(info[4][0]).split("%")[0]).is_global:
            return f"refusing {url}: {parsed.host} is not a public address"
    return None


async def read_capped(page: httpx.Response, limit: int) -> bytes:
    """At most `limit` bytes of a streamed body."""
    chunks, size = [], 0
    async for chunk in page.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit]


class Tools:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.transport: httpx.AsyncBaseTransport | None = None  # get_website's transport; tests swap in a mock

    def register(self, mcp: MCPServer) -> None:
        models = ", ".join(self.hub.settings.models)
        mcp.tool(description=(
            "Hire an agent. role is 'manager' (uses tools and may hire) or 'worker' (answers once, no tools). "
            f"model is one of: {models}. Reserves budget_usd from your pool."))(self.hire_agent)
        for tool in (self.fire_agent, self.check_usage, self.get_website, self.read_file, self.memory,
                     self.deliver_to_user):
            mcp.tool()(tool)

    def who(self, ctx: Context) -> tuple[Session, AgentRecord]:
        """The live session and agent behind the request's token. The auth gate has already rejected strangers;
        this also catches a token used after its agent was fired."""
        found = self.hub.caller(ctx.headers or {})
        rec = found[0].ledger.get(found[1]) if found else None
        if found is None or rec is None:
            raise PermissionError("unknown caller")
        if not rec.live:
            raise PermissionError(f"{rec.name} has been fired")
        return found[0], rec

    async def hire_agent(self, ctx: Context, name: str, role: str, model: str, prompt: str,
                         budget_usd: float) -> dict[str, Any]:
        session, boss = self.who(ctx)
        settings, depth = self.hub.settings, boss.depth + 1
        if boss.role != "manager":
            return {"error": "only managers can hire"}
        if not (math.isfinite(budget_usd) and budget_usd >= 0):
            return {"error": "budget_usd must be a finite number >= 0"}
        if model not in settings.models:
            return {"error": f"unknown model {model!r}; choose from {list(settings.models)}"}
        if role not in ROLES:
            return {"error": "role must be 'manager' or 'worker'"}
        if depth > settings.max_depth:
            return {"error": f"agents at depth {boss.depth} cannot hire (max depth {settings.max_depth})"}
        if not NAME.fullmatch(name) or name in RESERVED or session.ledger.get(name) is not None:
            return {"error": f"name {name!r} must be new, not {sorted(RESERVED)}, and use only a-z, 0-9 and _"}
        tier = "local" if is_local(model) else "cloud"
        rec = session.ledger.add(name, role, tier, model, prompt, boss.name, depth, budget_usd)  # claim the name
        if tier == "local":
            try:
                async with self.hub.model_lock(model):
                    rec.gpu_mb = await self.hub.ollama.load(model)
            except (httpx.HTTPError, KeyError, ValueError) as exc:  # Ollama down, or an unexpected reply
                session.ledger.agents.pop(name, None)
                return {"error": f"could not load {model} in Ollama: {exc}"}
            except BaseException:
                session.ledger.agents.pop(name, None)  # never leave a live record that holds money and a model
                raise
            if not rec.live or self.hub.sessions.get(session.id) is not session:
                # Fired, or the session closed, while the model loaded. The fire or close already counted this
                # agent and unloads its model once the load lock is free; it must not be mounted.
                return {"error": f"{name} was fired while {model} was loading"}
        self.hub.mount(session, rec)
        return {"hired": name, "tier": tier, "depth": depth, "reserved_usd": budget_usd, "gpu_mb": rec.gpu_mb}

    async def fire_agent(self, ctx: Context, name: str) -> dict[str, Any]:
        """Fire an agent you hired (directly or below you) and every agent it hired. Unspent money goes back."""
        session, me = self.who(ctx)
        target = session.ledger.get(name)
        if target is None or not target.live or me.name not in session.ledger.ancestors(name):
            return {"error": f"{name!r} is not a live agent below you"}
        fired = session.ledger.subtree(name)
        for n in fired:
            session.ledger.agents[n].live = False
            self.hub.unmount(session, n)
            self.hub.revoke(session, n)  # runs still in flight stop at their next LLM turn
        await self.hub.release_models({session.ledger.agents[n].model for n in fired})
        return {"fired": fired}

    async def check_usage(self, ctx: Context, pool: str) -> dict[str, Any]:
        """Money left in a pool (your pool is your own name), free memory in MB, and each local model's size in MB."""
        session, _ = self.who(ctx)
        if session.ledger.get(pool) is None:
            return {"error": f"no pool named {pool!r}"}
        return {"pool": pool, "api_usd": round(session.ledger.left()[pool], 6),
                "free_mb": free_mb(), "model_mb": await self.hub.ollama.sizes()}

    async def get_website(self, ctx: Context, url: str) -> dict[str, Any]:
        """Fetch a public web page and return its text."""
        self.who(ctx)
        if not url.startswith(("http://", "https://")):
            return {"error": "only http(s) URLs"}
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=False, transport=self.transport) as http:
                for _ in range(MAX_REDIRECTS + 1):  # follow redirects by hand so every hop is checked
                    if refused := await non_public(url):
                        return {"error": refused}
                    async with http.stream("GET", url) as page:
                        if page.is_redirect:
                            url = str(page.url.join(page.headers.get("location", "")))
                            continue
                        body = await read_capped(page, MAX_PAGE_BYTES)
                        text = body.decode(page.charset_encoding or "utf-8", errors="replace")
                        return {"status": page.status_code, "text": re.sub(r"<[^>]+>", " ", text)[:20000]}
        except httpx.HTTPError as exc:
            return {"error": f"could not fetch {url}: {exc}"}
        return {"error": f"more than {MAX_REDIRECTS} redirects"}

    def read_file(self, ctx: Context, path: str) -> dict[str, Any]:
        """Read a text file under the data directory, for example private/notes.md."""
        self.who(ctx)
        data = Path(self.hub.settings.data_dir).resolve()
        target = (data / path).resolve()
        if not target.is_relative_to(data) or not target.is_file():
            return {"error": f"no file {path} in the data directory"}
        return {"path": path, "text": target.read_text()}

    def memory(self, ctx: Context, action: str, key: str = "", value: str = "") -> dict[str, Any]:
        """Saved user memories. action 'get' returns all of them; 'add' saves value under key."""
        session, _ = self.who(ctx)
        if action == "add":
            session.memory[key] = value
            return {"saved": key}
        return {"memories": dict(session.memory)}

    def deliver_to_user(self, ctx: Context, text: str) -> dict[str, Any]:
        """Show text directly to the user."""
        self.who(ctx)
        return {"delivered": True}
