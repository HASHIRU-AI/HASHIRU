"""The live path of the purse command, with scripted models: one wallet per batch, stopping at max_usd, peers
required for outside agents, and HASHIRU and the LLM split run end to end."""
import asyncio
from dataclasses import replace

import pytest

from hashiru.hub import Hub
from hashiru.llm import WalletExceeded
from purse import cli
from purse.hashiru_run import HubMarket, bench_settings
from purse.jobs import WORKER_PROMPT, make_job
from purse.projects import Project
from purse.runner import RunConfig
from purse.table import OutcomeTable, Record
from tests.conftest import FakeOllama
from tests.test_purse_cli import write
from tests.test_purse_hashiru import CLOUD, TEST_MARKET, ScriptedLLM, call, reply


def scripted(script):
    """A HubFactory whose hubs think with `script` (one fresh copy per hub) and share the factory's wallet."""
    def make(settings, wallet):
        llm = ScriptedLLM(list(script))
        return Hub(settings, llm=llm, ollama=FakeOllama(), wallet=wallet)  # type: ignore[arg-type]
    return make


def test_live_hubs_share_one_wallet(tmp_path, fake_family):
    p = Project("p", (), 0.5, budget_usd=0.1)
    factory = cli.HubFactory(1.5)
    hubs = [factory(bench_settings(p, TEST_MARKET, out_dir=tmp_path, max_usd=1.5, ceo_model=CLOUD, port=9380 + i))
            for i in range(2)]
    assert hubs[0].wallet is hubs[1].wallet is factory.wallet and factory.wallet.max_usd == 1.5
    assert hubs[0].llm.wallet is factory.wallet


def test_hub_market_stops_when_the_wallet_is_spent(tmp_path, fake_family):
    job = make_job("fake", 1, "low", seed=1, uid="u")
    p = Project("p", (), 0.5, budget_usd=0.5)
    factory = cli.HubFactory(0.01, make=scripted([]))
    hub = factory(bench_settings(p, TEST_MARKET, out_dir=tmp_path, max_usd=0.01, ceo_model=CLOUD, port=9382))
    factory.wallet.spent = 0.02

    async def body():
        async with hub.serving(), HubMarket(hub, hub.new_session("cal", trace=False), reserve_usd=0.1) as market:
            return await market.attempt(job, CLOUD)
    with pytest.raises(BaseException) as info:  # the MCP client's task group may wrap it
        asyncio.run(body())
    assert cli.wallet_hit(info.value)


def test_wallet_hit_looks_inside_exception_groups():
    assert cli.wallet_hit(WalletExceeded("x")) and not cli.wallet_hit(RuntimeError("x"))
    assert cli.wallet_hit(ExceptionGroup("g", [RuntimeError("a"), ExceptionGroup("h", [WalletExceeded("x")])]))


def test_calibrate_refuses_outside_agents_without_a_peer(tmp_path, capsys):
    pool_file, _ = write(tmp_path)
    table = tmp_path / "new.jsonl"
    assert cli.main(["calibrate", "--pool", str(pool_file), "--table", str(table), "--options", "a2a/calc",
                     "--max-usd", "1", "--spend"]) == 2
    assert "--peer calc=" in capsys.readouterr().out and not table.exists()


def project_and_table(fake_family_job):
    job = replace(fake_family_job, id="j1", value=2)
    table = OutcomeTable(None)
    table.add(Record(job.uid, CLOUD, 0, f"<answer>{job.answer}</answer>", True, 0.003, 1.0))
    return Project("p", (job,), 0.5, budget_usd=0.5, memory_mb=1000), table


def test_live_hashiru_runs_end_to_end(tmp_path, fake_family):
    p, table = project_and_table(make_job("fake", 2, "low", seed=4, uid="fake-l2-low-000"))
    job = p.jobs[0]
    script = [reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": CLOUD,
                                               "prompt": WORKER_PROMPT, "budget_usd": 0.1}, "c1")]),
              reply(calls=[call("ask_agent", {"name": "w1", "text": job.prompt}, "c2")]),
              reply(f'{{"j1": "{job.answer}"}}')]
    cfg = RunConfig("hashiru", ceo=CLOUD, out=tmp_path, max_usd=1.0)
    factory = cli.HubFactory(1.0, make=scripted(script))
    extras, result = asyncio.run(cli._live(cfg, p, "do it", TEST_MARKET, 0, table, factory))
    assert result.answers == {"j1": job.answer} and result.allocator_usd == pytest.approx(0.003)
    assert extras["tried"] == {"j1": [CLOUD]} and extras["replay_score"] == 1.0 and extras["prompt_fidelity"] == 1.0


def test_live_llm_split_pays_for_its_plan_from_the_budget(tmp_path, fake_family):
    p, table = project_and_table(make_job("fake", 2, "low", seed=4, uid="fake-l2-low-000"))
    plan = {"summary": "s", "hires": [{"name": "j1", "role": "worker", "model": CLOUD, "budget_usd": 0.05,
                                       "task": "t"}]}
    cfg = RunConfig("llm-split", ceo=CLOUD, out=tmp_path, max_usd=1.0)
    factory = cli.HubFactory(1.0, make=scripted([reply(calls=[call("propose_plan", plan, "c1")]), reply("ok")]))
    extras, result = asyncio.run(cli._live(cfg, p, "plan it", TEST_MARKET, 0, table, factory))
    assert extras["allocation"] == {"j1": CLOUD} and result.allocator_usd == pytest.approx(0.002)
    assert result.spent_usd == pytest.approx(0.002 + 0.003) and set(result.answers) == {"j1"}


def test_live_runs_stop_once_the_wallet_is_spent(tmp_path, fake_family):
    p, table = project_and_table(make_job("fake", 2, "low", seed=4, uid="fake-l2-low-000"))
    factory = cli.HubFactory(0.01, make=scripted([]))
    factory.wallet.spent = 0.02
    with pytest.raises(WalletExceeded):
        asyncio.run(cli._live(RunConfig("hashiru", ceo=CLOUD, out=tmp_path), p, "x", TEST_MARKET, 0, table, factory))
