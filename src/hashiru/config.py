"""Settings: Hydra composes hashiru/conf/config.yaml, an optional user config file, and command-line overrides
(in that order of precedence, lowest first) into a frozen Settings."""
import math
import os
import re
from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlsplit

from hydra import compose, initialize_config_module
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

from hashiru.llm import is_local

__all__ = ["A2APeer", "McpServer", "Settings", "load_settings"]

SERVER_NAME = re.compile(r"[a-z][a-z0-9_]{0,23}")
SERVER_KEYS = {"command", "args", "env", "url", "headers", "labels", "egress"}
PEER_KEYS = {"url", "headers", "price_usd_per_message", "labels"}
POSITIVE = ("task_timeout", "tool_start_timeout", "tool_call_timeout", "mcp_start_timeout", "session_ttl",
            "max_sessions", "ollama_num_ctx")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
NON_NEGATIVE = ("budget_usd", "max_usd", "turn_reserve_usd")
LABELS = {"private", "untrusted"}


@dataclass(frozen=True)
class McpServer:
    """A third-party MCP server: a local `command` (stdio) or a remote `url` (streamable HTTP)."""

    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()       # passed to a stdio server on top of MCP's minimal environment
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()   # sent to an HTTP server, e.g. Authorization
    labels: tuple[str, ...] = ("untrusted",)    # what its results carry in the trace
    egress: bool = True                         # may send data off the machine (so private data may not reach it)


@dataclass(frozen=True)
class A2APeer:
    """A remote A2A agent managers may hire as ``a2a/<name>``. Its price comes from its Agent Card's cost
    extension, or from price_usd_per_message here; a peer with neither cannot be hired."""

    name: str
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    price_usd_per_message: float | None = None
    labels: tuple[str, ...] = ("untrusted",)  # what its replies carry in the trace


@dataclass(frozen=True)
class Settings:
    """Everything one HASHIRU process needs. See conf/config.yaml for what each field means."""

    ceo_model: str
    models: tuple[str, ...]
    budget_usd: float = 0.5
    max_usd: float = 2.0
    max_depth: int = 2
    max_turns: int = 12
    task_timeout: float = 1800.0  # seconds for one user request, across every agent it reaches
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_num_ctx: int = 16384  # context window for local models (Ollama's default, 4096, is too small)
    host: str = "127.0.0.1"
    port: int = 9200
    data_dir: str = "data"
    trace_dir: str = "traces"
    prices_file: str = ""
    tool_creation: bool = False
    tools_dir: str = "user_tools"
    tool_start_timeout: float = 180.0
    tool_call_timeout: float = 60.0
    mcp_start_timeout: float = 30.0
    mcp_servers: tuple[McpServer, ...] = ()
    memory_db: str = ""  # SQLite file shared by all sessions; empty: each session has its own copy of memory.json
    memory_autoinject: bool = False
    a2a_peers: tuple[A2APeer, ...] = ()
    enforce_budget: bool = True
    turn_reserve_usd: float = 0.01  # a cloud manager keeps this much for its own turns when hiring
    allow_hiring: bool = True
    allow_local_agents: bool = True
    allow_cloud_agents: bool = True
    api_key: str = field(default="", repr=False)  # bearer key for the public routes; required beyond loopback
    public_url: str = ""  # where others reach this hub, for the Agent Card; default http://<host>:<port>
    max_sessions: int = 16  # open conversations from the chat page, the OpenAI API, A2A and MCP clients
    session_ttl: float = 3600.0  # seconds an idle conversation stays open with its hires
    frame_ancestors: str = "'none'"  # CSP frame-ancestors for the chat page
    private_ceo_model: str = ""  # a private chat's CEO; default ceo_model if local, else the first local model
    private_egress: str = "block"  # "block": agents holding private data cannot call tools that leave the machine
    private_by_default: bool = False  # every chat is private, from its first message
    private_mode: str = "obfuscate"  # obfuscate: cloud models see private data masked; block: they never see it
    obfuscate_with: str = "patterns+model"  # patterns (Presidio) alone, or plus a local model for names and details
    obfuscate_model: str = ""  # the local model that finds them; default: as private_ceo_model

    @property
    def internal_url(self) -> str:
        """Where agents reach the hub. Always loopback, even when the hub listens on every interface."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def base_url(self) -> str:
        """Where clients reach the hub: public_url, or this machine's address."""
        if self.public_url:
            return self.public_url.rstrip("/")
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{f'[{host}]' if ':' in host else host}:{self.port}"

    @property
    def loopback(self) -> bool:
        return self.host in LOOPBACK


def _models(value: Any) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    return tuple(str(m).strip() for m in items if str(m).strip())


def _pairs(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{where} must map names to values")
    return tuple((str(k), str(v)) for k, v in value.items())


def _servers(value: Any) -> tuple[McpServer, ...]:
    if not value:
        return ()
    if not isinstance(value, dict):
        raise ValueError("mcp_servers must map server names to their settings")
    servers = []
    for raw_name, spec in value.items():
        name, where = str(raw_name), f"mcp_servers.{raw_name}"
        if not SERVER_NAME.fullmatch(name):
            raise ValueError(f"{where}: names use a-z, 0-9 and _ (at most 24)")
        if not isinstance(spec, dict) or set(map(str, spec)) - SERVER_KEYS:
            raise ValueError(f"{where} may only set {sorted(SERVER_KEYS)}")
        command, url = str(spec.get("command") or ""), str(spec.get("url") or "")
        if bool(command) == bool(url):
            raise ValueError(f"{where} needs exactly one of command (local) or url (remote)")
        args, labels = spec.get("args", []), spec.get("labels", ["untrusted"])
        if not isinstance(args, list) or not isinstance(labels, list):
            raise ValueError(f"{where}: args and labels must be lists")
        labels = tuple(str(label) for label in labels)
        if not set(labels) <= LABELS:
            raise ValueError(f"{where}.labels may only use {sorted(LABELS)}")
        egress = spec.get("egress", True)
        if not isinstance(egress, bool):
            raise ValueError(f"{where}.egress must be true or false")
        if url and not egress:
            raise ValueError(f"{where}: egress: false cannot hold for a server reached by url (it receives what "
                             "you send it)")
        servers.append(McpServer(name, command, tuple(str(a) for a in args),
                                 _pairs(spec.get("env"), f"{where}.env"), url,
                                 _pairs(spec.get("headers"), f"{where}.headers"), labels, egress))
    return tuple(servers)


def _peers(value: Any) -> tuple[A2APeer, ...]:
    if not value:
        return ()
    if not isinstance(value, dict):
        raise ValueError("a2a_peers must map peer names to their settings")
    peers = []
    for raw_name, spec in value.items():
        name, where = str(raw_name), f"a2a_peers.{raw_name}"
        if not SERVER_NAME.fullmatch(name):
            raise ValueError(f"{where}: names use a-z, 0-9 and _ (at most 24)")
        if not isinstance(spec, dict) or set(map(str, spec)) - PEER_KEYS:
            raise ValueError(f"{where} may only set {sorted(PEER_KEYS)}")
        url = str(spec.get("url") or "")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password \
                or parts.query or parts.fragment:
            raise ValueError(f"{where}.url must be a plain http(s) URL (put credentials in headers)")
        price = spec.get("price_usd_per_message")
        if price is not None and (isinstance(price, bool) or not isinstance(price, int | float)
                                  or not math.isfinite(price) or price < 0):
            raise ValueError(f"{where}.price_usd_per_message must be a finite number >= 0")
        labels = spec.get("labels", ["untrusted"])
        if not isinstance(labels, list) or not set(map(str, labels)) <= LABELS:
            raise ValueError(f"{where}.labels must be a list using {sorted(LABELS)}")
        peers.append(A2APeer(name, url, _pairs(spec.get("headers"), f"{where}.headers"),
                             None if price is None else float(price), tuple(map(str, labels))))
    return tuple(peers)


def _compose(overrides: list[str], config_file: str) -> DictConfig:
    with initialize_config_module(config_module="hashiru.conf", version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    if config_file:
        try:
            loaded = OmegaConf.load(config_file)
        except Exception as exc:  # missing file, YAML syntax error, ...: one clear message
            raise ValueError(f"cannot read {config_file}: {exc}") from exc
        merged = OmegaConf.merge(cfg, loaded)
        if not isinstance(merged, DictConfig):
            raise ValueError(f"{config_file} must be a YAML mapping")
        cfg = merged
    # Command-line overrides beat the file. They are applied to the merged config directly, so they can change
    # keys that exist only in the file, and their ${...} interpolations resolve against the merged result.
    for override in OverridesParser.create().parse_overrides(overrides):
        if override.is_delete() or override.is_sweep_override():
            raise ValueError(f"{override.input_line}: only key=value overrides are supported")
        OmegaConf.update(cfg, override.key_or_group, override.value(), merge=False, force_add=True)
    return cfg


def load_settings(overrides: list[str] | None = None, config_file: str | None = None) -> Settings:
    """Compose the config and return frozen Settings.

    `overrides` are Hydra overrides such as ``budget_usd=1.0``. `config_file` (default: $HASHIRU_CONFIG) is a
    YAML file with the same keys, e.g. for ``mcp_servers``.
    """
    raw = OmegaConf.to_container(
        _compose(overrides or [], config_file or os.environ.get("HASHIRU_CONFIG", "")), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("the config must be a mapping")
    known = {f.name: f.type for f in fields(Settings)}
    unknown = sorted(set(map(str, raw)) - set(known))
    if unknown:
        raise ValueError(f"unknown settings: {unknown}")
    values: dict[str, Any] = {}
    for raw_key, value in raw.items():
        key = str(raw_key)
        kind = known[key]
        if kind is bool and not isinstance(value, bool):  # "no" or "off" must not turn a switch on
            raise ValueError(f"{key} must be true or false, got {value!r}")
        if key == "models":
            values[key] = _models(value)
        elif key == "mcp_servers":
            values[key] = _servers(value)
        elif key == "a2a_peers":
            values[key] = _peers(value)
        else:
            try:
                values[key] = kind(value) if kind in (int, float, str) else value
            except (TypeError, ValueError):
                raise ValueError(f"{key} has an invalid value: {value!r}") from None
    for key in POSITIVE + NON_NEGATIVE:
        value = values.get(key)
        if value is not None and (not math.isfinite(value) or value < 0 or (key in POSITIVE and value == 0)):
            raise ValueError(f"{key} must be a finite number {'> 0' if key in POSITIVE else '>= 0'}, got {value}")
    if "api_key" in values:  # pasted secrets often carry a newline; clients' tokens are stripped too
        key = values["api_key"].strip()
        if values["api_key"] and not key:
            raise ValueError("api_key is blank")
        values["api_key"] = key
    settings = Settings(**values)
    if not settings.loopback and not settings.api_key:
        raise ValueError(f"listening on {settings.host} needs an api_key (set HASHIRU_API_KEY)")
    if settings.public_url and urlsplit(settings.public_url).scheme not in ("http", "https"):
        raise ValueError("public_url must be an http(s) URL")
    if settings.private_ceo_model and not is_local(settings.private_ceo_model):
        raise ValueError(f"private_ceo_model must be a local model (ollama_chat/...), got {settings.private_ceo_model}")
    if settings.private_egress not in ("block", "allow"):
        raise ValueError(f"private_egress must be 'block' or 'allow', got {settings.private_egress!r}")
    if settings.private_mode not in ("obfuscate", "block"):
        raise ValueError(f"private_mode must be 'obfuscate' or 'block', got {settings.private_mode!r}")
    if settings.obfuscate_with not in ("patterns+model", "patterns"):
        raise ValueError(f"obfuscate_with must be 'patterns+model' or 'patterns', got {settings.obfuscate_with!r}")
    if settings.obfuscate_model and not is_local(settings.obfuscate_model):
        raise ValueError(f"obfuscate_model must be a local model (ollama_chat/...), got {settings.obfuscate_model}")
    has_local = bool(settings.private_ceo_model or settings.obfuscate_model or is_local(settings.ceo_model)
                     or any(is_local(m) for m in settings.models))
    patterns_only = settings.private_mode == "obfuscate" and settings.obfuscate_with == "patterns"
    if settings.private_by_default and not (has_local or patterns_only):
        raise ValueError("private_by_default needs a local model: add one to models or set private_ceo_model")
    return settings
