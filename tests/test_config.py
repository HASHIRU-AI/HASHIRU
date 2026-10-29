"""Hydra config loading."""
import pytest

from hashiru.config import load_settings

ENV = ("CEO_MODEL", "MODELS", "BUDGET_USD", "MAX_USD", "OLLAMA_URL", "PORT", "HASHIRU_PRICES",
       "HASHIRU_TOOL_CREATION")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV:
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_no_keys():
    s = load_settings()
    assert s.ceo_model == "ollama_chat/qwen3.5:4b" and s.models == ("ollama_chat/qwen3.5:4b",)
    assert (s.budget_usd, s.max_usd, s.max_depth, s.port) == (0.5, 2.0, 2, 9100)
    assert s.internal_url == "http://127.0.0.1:9100"


def test_environment_variables(monkeypatch):
    monkeypatch.setenv("MODELS", "ollama_chat/a, gpt-4.1-mini ,")
    monkeypatch.setenv("BUDGET_USD", "1")
    monkeypatch.setenv("PORT", "9200")
    s = load_settings()
    assert s.models == ("ollama_chat/a", "gpt-4.1-mini")
    assert s.budget_usd == 1.0 and isinstance(s.budget_usd, float) and s.port == 9200


def test_command_line_overrides_win(monkeypatch):
    monkeypatch.setenv("BUDGET_USD", "1")
    s = load_settings(["budget_usd=0.25", "max_depth=3", 'models=["ollama_chat/a","gpt-4.1-mini"]'])
    assert (s.budget_usd, s.max_depth, s.models) == (0.25, 3, ("ollama_chat/a", "gpt-4.1-mini"))


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError, match="unknown settings"):
        load_settings(["+colour=blue"])


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_tool_creation_switch_is_a_real_boolean(monkeypatch, value, expected):
    assert load_settings().tool_creation is False
    monkeypatch.setenv("HASHIRU_TOOL_CREATION", value)
    assert load_settings().tool_creation is expected  # a string "false" would be truthy


@pytest.mark.parametrize("value", ["no", "off", "disabled", "1", "yes"])
def test_a_switch_accepts_only_true_or_false(monkeypatch, value):
    monkeypatch.setenv("HASHIRU_TOOL_CREATION", value)
    with pytest.raises(ValueError, match="true or false"):
        load_settings()


def test_a_quoted_false_on_the_command_line_is_refused():
    with pytest.raises(ValueError, match="true or false"):
        load_settings(['tool_creation="false"'])
