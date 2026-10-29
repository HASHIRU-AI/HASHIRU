"""Settings: Hydra composes hashiru/conf/config.yaml and command-line overrides into a frozen Settings."""
from dataclasses import dataclass, fields
from typing import Any

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

__all__ = ["Settings", "load_settings"]


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

    @property
    def internal_url(self) -> str:
        """Where agents reach the hub. Always loopback, even when the hub listens on every interface."""
        return f"http://127.0.0.1:{self.port}"


def _models(value: Any) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    return tuple(str(m).strip() for m in items if str(m).strip())


def load_settings(overrides: list[str] | None = None) -> Settings:
    """Compose the config with Hydra overrides such as ``budget_usd=1.0`` and return frozen Settings."""
    with initialize_config_module(config_module="hashiru.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("config.yaml must be a mapping")
    known = {f.name: f.type for f in fields(Settings)}
    unknown = sorted(set(map(str, raw)) - set(known))
    if unknown:
        raise ValueError(f"unknown settings: {unknown}")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        kind = known[str(key)]
        values[str(key)] = _models(value) if key == "models" else kind(value) if kind in (int, float, str) else value
    return Settings(**values)
