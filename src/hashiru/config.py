"""Settings: Hydra composes hashiru/conf/config.yaml, an optional user config file, and command-line overrides
(in that order of precedence, lowest first) into a frozen Settings."""
import os
import re
from dataclasses import dataclass, fields
from typing import Any

from hydra import compose, initialize_config_module
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

__all__ = ["McpServer", "Settings", "load_settings"]

SERVER_NAME = re.compile(r"[a-z][a-z0-9_]{0,23}")
SERVER_KEYS = {"command", "args", "env", "url", "headers", "labels"}
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


@dataclass(frozen=True)
class Settings:
    """Everything one HASHIRU process needs. See conf/config.yaml for what each field means."""

    ceo_model: str
    models: tuple[str, ...]
    budget_usd: float = 0.5
    max_usd: float = 2.0
    max_depth: int = 2
    max_turns: int = 12
    a2a_timeout: float = 600.0
    ollama_url: str = "http://127.0.0.1:11434"
    host: str = "127.0.0.1"
    port: int = 9100
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

    @property
    def internal_url(self) -> str:
        """Where agents reach the hub. Always loopback, even when the hub listens on every interface."""
        return f"http://127.0.0.1:{self.port}"


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
        servers.append(McpServer(name, command, tuple(str(a) for a in args),
                                 _pairs(spec.get("env"), f"{where}.env"), url,
                                 _pairs(spec.get("headers"), f"{where}.headers"), labels))
    return tuple(servers)


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
        else:
            try:
                values[key] = kind(value) if kind in (int, float, str) else value
            except (TypeError, ValueError):
                raise ValueError(f"{key} has an invalid value: {value!r}") from None
    return Settings(**values)
