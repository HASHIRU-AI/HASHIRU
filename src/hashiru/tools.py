"""The built-in MCP tools. Each one learns who is calling from the request's bearer token, never from
an argument, so an agent cannot act as another agent."""
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
from mcp.server.mcpserver import Context, MCPServer

from hashiru.ledger import AgentRecord, free_mb
from hashiru.llm import is_local
from hashiru.memory import MAX_KEY_CHARS
from hashiru.session import USER, Session
from hashiru.web import fetch_page, search_arxiv

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["ROLES", "Tools"]

ROLES = ("manager", "worker")
NAME = re.compile(r"[a-z0-9_]{1,32}")
RESERVED = {USER, "ceo"}  # an agent named "user" would get the human's token and speak to the CEO as them
ALWAYS = "ask_agent"      # the runtime's own tool, not served over MCP
CREATE_TOOL = (
    "Create a new tool that every manager can call from its next turn on. code is a Python module that "
    "defines a top-level function named exactly `name`, with a type hint on every parameter, a docstring "
    "(it becomes the tool's description) and a JSON-serializable return value. dependencies lists the PyPI "
    "packages the code imports, e.g. ['requests==2.32.3']. The tool runs in its own process with its own "
    "packages and cannot see this conversation; keep the docstring under 1000 characters. If the code does not "
    "load, nothing is created and the error says why, so fix the code and try again.")


class Tools:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.transport: httpx.AsyncBaseTransport | None = None  # web tools' transport; tests swap in a mock

    def register(self, mcp: MCPServer) -> None:
        models = ", ".join(self.hub.settings.models)
        mcp.tool(description=(
            "Hire an agent. role is 'manager' (uses tools and may hire) or 'worker' (answers once, no tools). "
            f"model is one of: {models}. Reserves budget_usd from your pool. For a manager, tools optionally "
            "limits which of your tools it may use (by name, including ask_agent)."))(self.hire_agent)
        for tool in (self.fire_agent, self.check_usage, self.get_website, self.arxiv_search, self.read_file,
                     self.memory, self.deliver_to_user):
            mcp.tool()(tool)
        if self.hub.settings.tool_creation:
            mcp.tool(description=CREATE_TOOL)(self.create_tool)
            mcp.tool()(self.delete_tool)

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
                         budget_usd: float, tools: list[str] | None = None) -> dict[str, Any]:
        session, boss = self.who(ctx)
        settings, depth = self.hub.settings, boss.depth + 1
        if boss.role != "manager":
            return {"error": "only managers can hire"}
        if tools is None and boss.tools is not None and role == "manager":
            tools = list(boss.tools)  # a limited manager cannot hire its way out of its limits
        if tools is not None:
            if role != "manager":
                return {"error": "tools can only be limited for a manager; workers have none"}
            mine = boss.tools if boss.tools is not None else \
                {t.name for t in await self.hub.mcp.list_tools()} | {ALWAYS}
            if not_mine := sorted(set(tools) - set(mine)):
                return {"error": f"you cannot grant tools you do not have: {not_mine}"}
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
        rec = session.ledger.add(name, role, tier, model, prompt, boss.name, depth, budget_usd,  # claim the name
                                 tools=tuple(sorted(set(tools))) if tools is not None else None)
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
        return await fetch_page(url, self.transport)

    async def arxiv_search(self, ctx: Context, query: str, max_results: int = 5) -> dict[str, Any]:
        """Search arXiv for papers: title, authors, date, abstract and PDF link for each (at most 20)."""
        self.who(ctx)
        return await search_arxiv(query, max_results, self.transport)

    def read_file(self, ctx: Context, path: str) -> dict[str, Any]:
        """Read a text file under the data directory, for example private/notes.md."""
        self.who(ctx)
        data = Path(self.hub.settings.data_dir).resolve()
        target = (data / path).resolve()
        if not target.is_relative_to(data) or not target.is_file():
            return {"error": f"no file {path} in the data directory"}
        return {"path": path, "text": target.read_text()}

    def memory(self, ctx: Context, action: Literal["search", "add", "delete", "list"], query: str = "",
               key: str = "", text: str = "", limit: int = 5) -> dict[str, Any]:
        """The user's saved memories (preferences, facts, past solutions). 'search' returns the memories that best
        match query (the most recent if query is empty); 'add' saves text under key, replacing it; 'delete'
        removes key; 'list' returns every key."""
        session, me = self.who(ctx)
        store = session.memory
        if len(key) > MAX_KEY_CHARS:
            return {"error": f"keys are at most {MAX_KEY_CHARS} characters"}
        if action == "add":
            if not key or not text:
                return {"error": "add needs key and text"}
            store.add(key, text, sorted(session.trace.labels(me.name, session.id)))  # the writer's labels stay
            return {"saved": key}
        if action == "delete":
            return {"deleted": key} if store.delete(key) else {"error": f"no memory named {key!r}"}
        if action == "list":
            keys, total = store.keys()
            return {"keys": keys, "total": total} | ({"note": f"the {len(keys)} newest"} if total > len(keys) else {})
        hits = store.search(query, max(1, min(limit, 20)))
        # _labels is read (and removed) by the runtime: what was written under untrusted stays untrusted.
        return {"memories": [{"key": h["key"], "text": h["text"]} for h in hits],
                "_labels": sorted({label for h in hits for label in h["labels"]})}

    def deliver_to_user(self, ctx: Context, text: str) -> dict[str, Any]:
        """Show text directly to the user."""
        self.who(ctx)
        return {"delivered": True}

    async def create_tool(self, ctx: Context, name: str, code: str,
                          dependencies: list[str] | None = None) -> dict[str, Any]:
        session, me = self.who(ctx)
        return await self.hub.toolsmith.create(name, code, list(dependencies or []), creator=f"{session.id}/{me.name}")

    async def delete_tool(self, ctx: Context, name: str) -> dict[str, Any]:
        """Delete a tool your session made with create_tool."""
        session, _ = self.who(ctx)
        return await self.hub.toolsmith.delete(name, session)
