"""One process hosts every session: MCP tools at /mcp, and one A2A endpoint per hired agent at
/s/{session}/agents/{name}/, all behind a bearer-token gate. The same port serves the public routes
(chat page, OpenAI-compatible API, public A2A agent and MCP server; see hashiru.interop)."""
import asyncio
import contextlib
import logging
import re
import secrets
import socket
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.routing import Mount

from hashiru.auth import AuthGate, token_of
from hashiru.config import AUTO, LOOPBACK, A2APeer, Settings
from hashiru.gateway import Gateway
from hashiru.interop import Conversations, mount
from hashiru.ledger import AgentRecord, free_mb
from hashiru.llm import LLM, Wallet, check_priced, is_local, register_prices
from hashiru.memory import MemoryStore, open_memory, words
from hashiru.obfuscation import Detector, Obfuscator
from hashiru.ollama import Ollama, tag
from hashiru.peers import PEER_PREFIX, BoundedTaskStore, PeerExecutor, Price, cost_extension, model_price
from hashiru.privacy import EGRESS, RULE, holds_private, may_hold, refuse
from hashiru.recon import use_cached_map
from hashiru.router import Router
from hashiru.runtime import CEO_PROMPT, Executor
from hashiru.session import USER, Session
from hashiru.tools import Tools
from hashiru.toolsmith import Toolsmith
from hashiru.trace import PRIVATE, SOURCES, UNTRUSTED, Trace

__all__ = ["Hub"]

log = logging.getLogger(__name__)
LABEL = re.compile(r"[a-z0-9_]{1,32}")


class Hub:
    def __init__(self, settings: Settings, llm: LLM | None = None, ollama: Ollama | None = None) -> None:
        if settings.models == AUTO:
            raise ValueError("models=auto must be settled first (hashiru.recon.settle)")
        use_cached_map(settings.model_catalog, [*settings.models, settings.ceo_model])  # a refresh's newer prices
        if settings.prices_file:
            register_prices(Path(settings.prices_file))
        check_priced([*settings.models, settings.ceo_model])
        self.settings = settings
        self.wallet = Wallet(settings.max_usd)
        self.llm = llm or LLM(settings.ollama_url, self.wallet, settings.ollama_num_ctx)
        known = (*settings.models, settings.ceo_model, settings.private_ceo_model)  # Ollama sizes its local ones
        self.ollama = ollama or Ollama(settings.ollama_url, known, settings.ollama_num_ctx)
        if (urlsplit(settings.ollama_url).hostname or "") not in LOOPBACK:
            log.warning("ollama_url %s is not this machine: local models, and the private data they see, run there",
                        settings.ollama_url)
        self.sessions: dict[str, Session] = {}
        self._tokens: dict[str, tuple[str, str]] = {}  # token -> (session id, caller name)
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._running: dict[tuple[str, str], set[asyncio.Task[Any]]] = {}  # (session, agent) -> its A2A runs
        self._agent_locks: dict[tuple[str, str], asyncio.Lock] = {}  # one request at a time per agent
        self.memory_lock = asyncio.Lock()  # local hires check free memory, reserve and load one at a time
        self.peer_cards: dict[tuple[str, str], AgentCard] = {}  # a hired peer's card, fetched once and pinned
        self._background: set[asyncio.Task[None]] = set()
        self._server: asyncio.Task[None] | None = None
        self.mcp = MCPServer("hashiru")
        self.tools = Tools(self)
        self.tools.register(self.mcp)
        self.toolsmith = Toolsmith(self)
        self.gateway = Gateway(self)
        self.memory: MemoryStore | None = (  # shared by all sessions when persistent
            open_memory(settings.memory_db, Path(settings.data_dir)) if settings.memory_db else None)
        self.app = self.mcp.streamable_http_app(host=settings.host)
        self._obfuscator = self._make_obfuscator()
        self.router = Router(self)
        self.conversations = Conversations(self)
        self.public_mcp = mount(self)
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
                          memory=self.memory or open_memory("", Path(s.data_dir)))
        self.sessions[sid] = session
        ceo = session.ledger.add("ceo", "manager", tier, s.ceo_model, CEO_PROMPT, None, 0, s.budget_usd)
        self.issue(session, USER)
        self.mount(session, ceo)
        return session

    async def close_session(self, session: Session) -> None:
        """Take down every agent, revoke the session's tokens and unload models nobody uses anymore.

        Runs still in flight are cancelled.
        """
        self.sessions.pop(session.id, None)
        session.approvals.drop()  # nobody will answer its questions now
        for rec in session.ledger.agents.values():
            rec.live = False
            self.unmount(session, rec.name)
        await self.stop_runs(session, list(session.ledger.agents))  # before revoking: runs still close their MCP
        for name in list(session.tokens):
            self.revoke(session, name)
        for name in list(session.ledger.agents):
            self.peer_cards.pop((session.id, name), None)
            self._agent_locks.pop((session.id, name), None)
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
        price = Price(per_message_usd=rec.price_usd) if rec.model.startswith(PEER_PREFIX) else model_price(rec.model)
        card = AgentCard(
            name=rec.name, description=f"{rec.role} on {rec.model}", version="1",
            default_input_modes=["text/plain"], default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=False, extensions=[cost_extension(price)]),
            supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
            skills=[AgentSkill(id=rec.role, name=rec.role, description=rec.role,
                               tags=[f"tier:{rec.tier}", f"depth:{rec.depth}"])])
        executor: AgentExecutor = (PeerExecutor(self, session, rec.name) if rec.model.startswith(PEER_PREFIX)
                                   else Executor(self, session, rec.name))
        handler = DefaultRequestHandler(agent_executor=executor, task_store=BoundedTaskStore(), agent_card=card)
        routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
        self.app.router.routes.append(Mount(self._path(session, rec.name), app=Starlette(routes=routes)))

    # ---- running agents ----

    def track(self, session: Session, name: str, task: asyncio.Task[Any] | None) -> None:
        if task is not None:
            self._running.setdefault((session.id, name), set()).add(task)

    def untrack(self, session: Session, name: str, task: asyncio.Task[Any] | None) -> None:
        runs = self._running.get((session.id, name))
        if runs is not None and task is not None:
            runs.discard(task)
            if not runs:
                del self._running[(session.id, name)]

    async def stop_runs(self, session: Session, names: list[str], wait: float = 5.0) -> None:
        """Cancel what fired agents (or a closed session's agents) are doing, and give the runs a moment to clean up
        (close their MCP sessions, answer their callers)."""
        me = asyncio.current_task()
        tasks = [t for name in names for t in self._running.get((session.id, name), ()) if t is not me]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=wait)

    def agent_lock(self, session: Session, name: str) -> asyncio.Lock:
        return self._agent_locks.setdefault((session.id, name), asyncio.Lock())

    # ---- models and peers ----

    def peer(self, model: str) -> A2APeer | None:
        name = model.removeprefix(PEER_PREFIX) if model.startswith(PEER_PREFIX) else None
        return next((p for p in self.settings.a2a_peers if p.name == name), None)

    def hireable(self) -> list[str]:
        """Models a manager may hire with, under allow_local_agents / allow_cloud_agents (peers count as cloud)."""
        s = self.settings
        models = [m for m in s.models if (s.allow_local_agents if is_local(m) else s.allow_cloud_agents)]
        return models + ([f"{PEER_PREFIX}{p.name}" for p in s.a2a_peers] if s.allow_cloud_agents else [])

    def unmount(self, session: Session, name: str) -> None:
        path = self._path(session, name)
        self.app.router.routes[:] = [r for r in self.app.router.routes if getattr(r, "path", None) != path]

    @staticmethod
    def _path(session: Session, name: str) -> str:
        return f"/s/{session.id}/agents/{name}"

    def sources(self) -> dict[str, list[str]]:
        """Labels each tool's results carry. Agent-created tools may fetch anything, so they are untrusted;
        a configured MCP server's tools carry the labels its config gives (untrusted by default)."""
        return SOURCES | self.gateway.labels() | {name: [UNTRUSTED] for name in self.toolsmith.names()}

    def recall(self, session: Session, text: str, limit: int = 5) -> str:
        """`text` plus the saved memories that best match it, when memory_autoinject is on (v1's behavior).

        Clients call this on the user's own words, before adding chat history. The CEO then holds private data,
        so its label set records that, and a 'recall' event records what was added. A cloud CEO gets them only
        obfuscated: the user is then private, so the message is masked on its way to the CEO; in block mode it gets
        nothing (R3).
        """
        if not self.settings.memory_autoinject or not words(text):  # no words: nothing to match on
            return text
        if not (hits := session.memory.search(text, limit)):
            return text
        labels = sorted({PRIVATE} | {lbl for h in hits for lbl in h["labels"]})
        local_ceo = may_hold(session.ledger.agents["ceo"].tier)
        if not local_ceo and self.obfuscating() is None:
            refuse(session, "ceo", "recall", "saved memories are private and the CEO runs on a cloud model")
            return text
        session.trace.emit("recall", task=session.id, agent="ceo", query=text[:200], results=hits)
        # A local CEO holds them; for a cloud one the user's message carries them, so it is masked on the way.
        session.trace.receive("ceo" if local_ceo else USER, session.id, labels)
        lines = "\n".join(f"- {h['key']}: {h['text']}" for h in hits)
        return f"{text}\n\nSaved memories that may be relevant:\n{lines}"

    def egress(self) -> set[str]:
        """Tools that may send data off this machine, which an agent holding private data may not call (R3): the web
        tools, tool creation, created tools (they may fetch anything) and configured MCP servers not marked
        egress: false."""
        return set(EGRESS) | self.toolsmith.names() | self.gateway.egress()

    def private_ceo_model(self) -> str | None:
        """The local model a private chat's CEO runs on: private_ceo_model, else ceo_model if it is local, else the
        first local model in models."""
        s = self.settings
        if s.private_ceo_model:
            return s.private_ceo_model
        return s.ceo_model if is_local(s.ceo_model) else next((m for m in s.models if is_local(m)), None)

    async def fits(self, model: str) -> str | None:
        """Why a local model would not fit in free memory now, or None. Callers hold memory_lock. A model that a live
        agent already uses is loaded, so it fits."""
        if not self.settings.enforce_budget or self.in_use(model):
            return None
        need, free = (await self.ollama.sizes()).get(tag(model), 0), free_mb()
        return f"{model} needs {need} MB of memory and only {free} MB is free" if need > free else None

    def _make_obfuscator(self) -> Obfuscator | None:
        s = self.settings
        if s.private_mode != "obfuscate":
            return None
        if s.obfuscate_with == "patterns":
            return Obfuscator(Detector(None, ""))
        model = s.obfuscate_model or self.private_ceo_model()
        if model is None:
            log.warning("private_mode is obfuscate but no local model can find names for it: private data is "
                        "blocked from cloud models instead (add a local model, or set obfuscate_with=patterns)")
            return None
        return Obfuscator(Detector(self.llm, model))

    def obfuscating(self) -> Obfuscator | None:
        """How private data reaches cloud models: through this obfuscator, or (None) not at all."""
        return self._obfuscator

    async def make_private(self, session: Session) -> str | None:
        """Make the rest of this session private: everything the user says carries `private`. When obfuscating, the
        CEO stays where it is and sees it obfuscated; otherwise it must run on private_ceo_model and moves to it
        (also after a delivery made the user private while the CEO was in the cloud). Returns why that is
        impossible, or None. Calling it again changes nothing."""
        ceo, target = session.ledger.agents["ceo"], self.private_ceo_model()
        if self.obfuscating() is not None:
            if not holds_private(session, USER):
                session.trace.receive(USER, session.id, [PRIVATE])
                session.trace.emit("policy", task=session.id, agent="ceo", rule=RULE, action="private",
                                   model=ceo.model, tier=ceo.tier, why="the chat is private from here on; cloud "
                                   "models see it obfuscated")
            return None
        if holds_private(session, USER) and ceo.model == target:
            return None
        if target is None:
            return self._private_failed(session, "this hub has no local model for private chats: add one to models "
                                                 "or set private_ceo_model")
        if not is_local(target):
            return self._private_failed(session, f"private_ceo_model {target} is not a local model")
        if ceo.model != target:
            try:
                async with self.memory_lock:  # the same check a local hire makes
                    if why := await self.fits(target):
                        return self._private_failed(session, f"{why}, so this chat cannot go private")
                    ceo.model, ceo.tier = target, "local"
            except httpx.HTTPError as exc:
                return self._private_failed(session, f"could not reach Ollama to check memory for {target}: {exc}")
            self.unmount(session, "ceo")
            self.mount(session, ceo)  # its card names the new model
        session.trace.receive(USER, session.id, [PRIVATE])
        session.trace.emit("policy", task=session.id, agent="ceo", rule=RULE, action="private", model=ceo.model,
                           tier=ceo.tier, why=f"the chat is private from here on; the CEO runs on {ceo.model}")
        return None

    @staticmethod
    def _private_failed(session: Session, why: str) -> str:
        session.trace.emit("policy", task=session.id, agent="ceo", rule=RULE, action="private_failed", why=why)
        return why

    def external(self) -> set[str]:
        """Tools whose error text comes from outside the hub, so their errors keep their labels too."""
        return self.toolsmith.names() | self.gateway.names()

    # ---- local models ----

    def in_use(self, model: str) -> bool:
        return any(a.live and a.model == model for s in self.sessions.values() for a in s.ledger.agents.values())

    def model_lock(self, model: str) -> asyncio.Lock:
        """Serializes loading and unloading one local model, so an unload never undoes a concurrent hire."""
        return self._model_locks.setdefault(model, asyncio.Lock())

    def release_soon(self, models: set[str]) -> None:
        """release_models from a path that cannot await (e.g. a cancelled hire)."""
        task = asyncio.get_running_loop().create_task(self.release_models(models))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

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
        if not self.settings.loopback and not self.settings.api_key:
            raise RuntimeError(f"listening on {self.settings.host} needs an api_key (set HASHIRU_API_KEY)")
        # uvicorn exits the whole process (SystemExit inside its task) when it cannot bind, so check first.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((self.settings.host, self.settings.port))
            except OSError as exc:
                raise RuntimeError(f"cannot listen on {self.settings.host}:{self.settings.port} ({exc.strerror}); "
                                   "pass port=<another>") from None
        srv = uvicorn.Server(uvicorn.Config(self.asgi, host=self.settings.host, port=self.settings.port,
                                            log_level="warning", timeout_graceful_shutdown=5))
        async with self.public_mcp.server.session_manager.run():
            task = asyncio.create_task(srv.serve())
            self._server = task
            while not srv.started:
                if task.done():
                    await task  # surfaces a startup failure
                    raise RuntimeError("server stopped during startup")
                await asyncio.sleep(0.05)
            sweeper = asyncio.create_task(self.conversations.sweeper())  # idle conversations expire on time
            try:
                await self.gateway.start()
                if self.settings.tool_creation:
                    await self.toolsmith.start()
                yield
            finally:
                sweeper.cancel()
                await self.conversations.close_all()
                await self.toolsmith.stop()
                await self.gateway.stop()
                srv.should_exit = True
                await task

    async def until_stopped(self) -> None:
        """Wait (inside serving()) until the server stops, e.g. on Ctrl-C."""
        if self._server is not None:
            await asyncio.shield(self._server)
