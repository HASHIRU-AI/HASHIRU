"""The open-weight market: models served by vLLM on rented GPUs and reached with --endpoint, priced per token at hosted
list prices from bench/prices.json, so a pool pays for them as it would through a provider (no network, no model
calls)."""
import contextlib

import litellm
import pytest

from hashiru.llm import affordable, has_price, register_prices
from purse import cli
from purse.hashiru_run import PRICES, bench_settings
from purse.market import ALL_OPTIONS, OPEN, calibrated
from purse.projects import Project
from tests.test_purse_cli import write

URL = "http://127.0.0.1:8001/v1"
MAX_OUTPUT = 8192  # Qwen3.5 thinks at temperature 0 until it is cut off: the pilot caps that (2026-11-02)


def test_each_open_model_is_charged_its_list_price():
    register_prices(PRICES)
    for o in OPEN:
        assert o.kind == "cloud" and o.name.startswith("hosted_vllm/") and has_price(o.name)
        served = litellm.ModelResponse(model=o.name.removeprefix("hosted_vllm/"),  # what vLLM names in its reply
                                       usage=litellm.Usage(prompt_tokens=1000, completion_tokens=1000,
                                                           total_tokens=2000))
        served._hidden_params = {"custom_llm_provider": "hosted_vllm"}
        assert litellm.completion_cost(completion_response=served) == pytest.approx(
            (o.input_usd_per_m + o.output_usd_per_m) / 1000)
        # a pool with plenty left still asks for no more than vLLM's context leaves after the longest input
        assert affordable(o.name, [{"role": "user", "content": "x"}], None, 10.0)[0] == MAX_OUTPUT


def test_open_models_join_a_market_only_once_calibrated():
    assert set(OPEN) <= set(ALL_OPTIONS)
    assert [o.name for o in calibrated({"gpt-6-luna", "a2a/csp"})] == ["gpt-6-luna", "a2a/csp"]
    assert calibrated({OPEN[0].name}) == (OPEN[0],)


def test_endpoints_reach_the_hub(tmp_path):
    p = Project("p", (), 1.0, budget_usd=0.5)
    s = bench_settings(p, OPEN, out_dir=tmp_path, max_usd=1.0, endpoints={OPEN[0].name: URL})
    assert s.endpoints == ((OPEN[0].name, URL),) and s.models == tuple(o.name for o in OPEN)
    assert bench_settings(p, OPEN, out_dir=tmp_path, max_usd=1.0).endpoints == ()


def test_calibrating_an_open_model_needs_its_endpoint(tmp_path, capsys, monkeypatch):
    pool_file, _ = write(tmp_path)
    base = ["calibrate", "--pool", str(pool_file), "--table", str(tmp_path / "new.jsonl"), "--levels", "1",
            "--per-cell", "1", "--max-usd", "1", "--options", OPEN[0].name]
    assert cli.main(base) == 2 and f"--endpoint {OPEN[0].name}=URL" in capsys.readouterr().out
    assert cli.main([*base, "--endpoint", f"{OPEN[0].name}={URL}"]) == 2  # no --spend: it only counts
    assert "12 calls" in capsys.readouterr().out
    seen = {}

    @contextlib.asynccontextmanager
    async def hub_market(*args, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")
        yield
    monkeypatch.setattr(cli, "_hub_market", hub_market)
    with pytest.raises(RuntimeError, match="stop here"):
        cli.main([*base, "--endpoint", f"{OPEN[0].name}={URL}", "--spend"])
    assert seen["endpoints"] == {OPEN[0].name: URL}


def test_an_endpoint_must_name_a_model_and_a_url():
    assert cli._endpoints([f"{OPEN[0].name}={URL}"]) == {OPEN[0].name: URL}
    for bad in ("no-equals-sign", f"={URL}", f"{OPEN[0].name}="):
        with pytest.raises(SystemExit):
            cli._endpoints([bad])


def test_the_hubs_a_command_builds_send_open_models_to_their_servers(tmp_path):
    """The pilot's first run reached no server: the factory built its LLM without the endpoints."""
    p = Project("p", (), 1.0, budget_usd=0.5)
    settings = bench_settings(p, OPEN, out_dir=tmp_path, max_usd=1.0, endpoints={OPEN[0].name: URL})
    hub = cli.HubFactory(1.0)(settings)
    assert hub.llm.endpoints == {OPEN[0].name: URL}
