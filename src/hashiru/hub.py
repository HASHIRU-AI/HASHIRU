"""One process hosts every session: MCP tools at /mcp, and one A2A endpoint per hired agent at
/s/{session}/agents/{name}/, all behind a bearer-token gate."""
import asyncio
import contextlib
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import httpx
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.routing import Mount

from hashiru.auth import AuthGate, token_of
from hashiru.config import Settings
from hashiru.ledger import AgentRecord
from hashiru.llm import LLM, Wallet, check_priced, is_local, register_prices
from hashiru.ollama import Ollama
from hashiru.runtime import CEO_PROMPT, Executor
from hashiru.session import USER, Session
from hashiru.tools import Tools, load_memory
from hashiru.trace import Trace

__all__ = ["Hub"]

log = logging.getLogger(__name__)
LABEL = re.compile(r"[a-z0-9_]{1,32}")


class Hub:
    def __init__(self, settings: Settings, llm: LLM | None = None, ollama: Ollama | None = None) -> None:
        if settings.prices_file:
            register_prices(Path(settings.prices_file))
        check_priced([*settings.models, settings.ceo_model])
        self.settings = settings
        self.wallet = Wallet(settings.max_usd)
        self.llm = llm or LLM(settings.ollama_url, self.wallet)
        self.ollama = ollama or Ollama(settings.ollama_url, settings.models)
        self.sessions: dict[str, Session] = {}
        self._tokens: dict[str, tuple[str, str]] = {}  # token -> (session id, caller name)
        self._model_locks: dict[str, asyncio.Lock] = {}
        self.mcp = MCPServer("hashiru")
        self.tools = Tools(self)
        self.tools.register(self.mcp)
        self.app = self.mcp.streamable_http_app(host=settings.host)
        self.asgi = AuthGate(self.app, self)

    # ---- sessions ----

    def new_session(self, label: str = "session", trace: bool = True) -> Session:
        """A fresh CEO holding the whole budget, its trace, and a token for the user."""
        if not LABEL.fullmatch(label):
            raise ValueError(f"session label {label!r} must use only a-z, 0-9 and _ (at most 32)")
        s = self.settings
        for _ in range(8):  # the id is a URL segment and a file name: it must be unique
            sid = f"{label}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(6)}"
            if sid not in self.sessions:
                break
        else:
            raise RuntimeError("could not draw an unused session id")
        tier = "local" if is_local(s.ceo_model) else "cloud"
        path = Path(s.trace_dir) / f"{sid}.jsonl" if trace else None
        session = Session(sid, Trace(path, task=sid, budget_usd=s.budget_usd, models=list(s.models),
                                     agents={"ceo": {"tier": tier, "depth": 0, "model": s.ceo_model}}),
                          memory=load_memory(Path(s.data_dir)))
        self.sessions[sid] = session
        ceo = session.ledger.add("ceo", "manager", tier, s.ceo_model, CEO_PROMPT, None, 0, s.budget_usd)
        self.issue(session, USER)
        self.mount(session, ceo)
        return session

    async def close_session(self, session: Session) -> None:
        """Take down every agent, revoke the session's tokens and unload models nobody uses anymore.

        Runs still in flight notice at their next LLM turn (runtime.still_on) and stop.
        """
        self.sessions.pop(session.id, None)
        for rec in session.ledger.agents.values():
            rec.live = False
            self.unmount(session, rec.name)
        for name in list(session.tokens):
            self.revoke(session, name)
        await self.release_models({rec.model for rec in session.ledger.agents.values()})

    # ---- identity ----

    def issue(self, session: Session, name: str) -> str:
        token = secrets.token_urlsafe(24)
        session.tokens[name] = token
        self._tokens[token] = (session.id, name)
        return token

    def revoke(self, session: Session, name: str) -> None:
        token = session.tokens.pop(name, None)
        if token:
            self._tokens.pop(token, None)

    def identity(self, token: str) -> tuple[Session, str] | None:
        sid, name = self._tokens.get(token, ("", ""))
        session = self.sessions.get(sid)
        return (session, name) if session else None

    def caller(self, headers: Mapping[str, str]) -> tuple[Session, str] | None:
        token = token_of(headers)
        return self.identity(token) if token else None

    # ---- A2A hosting ----

    def agent_url(self, session: Session, name: str) -> str:
        return f"{self.settings.internal_url}/s/{session.id}/agents/{name}/"

    def mount(self, session: Session, rec: AgentRecord) -> None:
        """Publish an agent as its own A2A service (card + JSON-RPC). Managers get a token to call tools."""
        if rec.role == "manager" and rec.name not in session.tokens:
            self.issue(session, rec.name)
        url = self.agent_url(session, rec.name)
        card = AgentCard(
            name=rec.name, description=f"{rec.role} on {rec.model}", version="1",
            default_input_modes=["text/plain"], default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=False),
            supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
            skills=[AgentSkill(id=rec.role, name=rec.role, description=rec.role,
                               tags=[f"tier:{rec.tier}", f"depth:{rec.depth}"])])
        handler = DefaultRequestHandler(agent_executor=Executor(self, session, rec.name),
                                        task_store=InMemoryTaskStore(), agent_card=card)
        routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
        self.app.router.routes.append(Mount(self._path(session, rec.name), app=Starlette(routes=routes)))

    def unmount(self, session: Session, name: str) -> None:
        path = self._path(session, name)
        self.app.router.routes[:] = [r for r in self.app.router.routes if getattr(r, "path", None) != path]

    @staticmethod
    def _path(session: Session, name: str) -> str:
        return f"/s/{session.id}/agents/{name}"

    # ---- local models ----

    def in_use(self, model: str) -> bool:
        return any(a.live and a.model == model for s in self.sessions.values() for a in s.ledger.agents.values())

    def model_lock(self, model: str) -> asyncio.Lock:
        """Serializes loading and unloading one local model, so an unload never undoes a concurrent hire."""
        return self._model_locks.setdefault(model, asyncio.Lock())

    async def release_models(self, models: set[str]) -> None:
        """Unload each local model that no live agent in any session still uses.

        in_use is checked under the model's lock right before each unload: an earlier unload may have
        yielded to a hire of the next model.
        """
        for model in sorted(m for m in models if is_local(m)):
            async with self.model_lock(model):
                if self.in_use(model):
                    continue
                try:
                    await self.ollama.unload(model)
                except httpx.HTTPError as exc:
                    log.warning("could not unload %s: %s", model, exc)

    # ---- serving ----

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        """Serve the hub inside the current event loop."""
        srv = uvicorn.Server(uvicorn.Config(self.asgi, host=self.settings.host, port=self.settings.port,
                                            log_level="warning"))
        task = asyncio.create_task(srv.serve())
        while not srv.started:
            if task.done():
                await task  # surfaces a startup failure such as the port already being in use
                raise RuntimeError("server stopped during startup")
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            srv.should_exit = True
            await task
