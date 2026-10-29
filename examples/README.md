# Examples

| File | What it is |
|---|---|
| [`hashiru.yaml`](hashiru.yaml) | A config file: models, two MCP servers (`fetch`, `clock`), a remote A2A agent (`calc`), and, commented out, another hub and a hosted agent with a key |
| [`clock_mcp_server.py`](clock_mcp_server.py) | An MCP server that tells agents the date and time (`clock__now`, `clock__days_between`) |
| [`calc_a2a_agent.py`](calc_a2a_agent.py) | A remote A2A agent that does exact arithmetic for a price, declared through the [cost extension](../docs/a2a-cost-extension.md) |

The two scripts list their own dependencies (inline script metadata), so `uv run` starts each in its own
environment, separate from HASHIRU's; any MCP host or A2A client can use them, not only HASHIRU. They need no LLM
and no API key. `tests/test_examples.py` runs both through a hub and checks that their pins match the versions
HASHIRU is tested with, so they keep working as the code changes.

## Try them with a hub

Run these from the repository root (the config refers to `examples/clock_mcp_server.py` by a relative path). The
first `uv run` of each script downloads its dependencies; if the hub logs that the clock server did not start in
time, restart the hub with `mcp_start_timeout=120` on its command line.

```bash
uv run examples/calc_a2a_agent.py --port 9300           # terminal 1: the remote agent, $0.001 per message
uv run hashiru serve --config examples/hashiru.yaml     # terminal 2: the hub starts the clock server itself
```

Managers now see `clock__now` and `clock__days_between` among their tools, and `a2a/calc` among the models they
can hire (the config lists calc whether or not it is running; a hire of it fails with "could not reach calc" if it
is not). Things to ask: *"How many days are left until 2026-12-25? Use the clock."* or *"Hire the calc agent and
have it work out (17.5 / 100) * 2340."* calc answers with the expression it read, `(17.5 / 100) * 2340 = 409.5`,
so a manager can see when it misread the question. The Agents panel shows the hire, its pool and what each message
cost.

## Make your own

**An MCP server** is any program that speaks MCP over stdio or streamable HTTP. Add it under `mcp_servers`; its
tools reach agents as `<server>__<tool>`. A local server gets MCP's minimal environment plus only the `env` you
list, and its results are labeled `untrusted` in the trace unless you set `labels`. Raise the SDK's `ToolError`
for failures the agent should read; any other exception reaches it only as "Error executing tool".

**An A2A agent** is hireable once the hub knows its price: declare `per_message_usd` (or token prices) through the
[cost extension](../docs/a2a-cost-extension.md) on its Agent Card, or set `price_usd_per_message` in the hub's
config. The hub reads the card from `<url>/.well-known/agent-card.json`, sends messages only to `<url>`, passes
the request's deadline as `hashiru_deadline`, and charges the larger of the price and the `hashiru_cost_usd` the
reply reports. Answer in task mode (complete the request's task, as `calc_a2a_agent.py` does) rather than with a
bare message.

## Two hubs

A HASHIRU hub is itself an A2A agent at `<base_url>/a2a`, so one hub can hire another as a worker: the other
hub's CEO, and every agent it hires, answer as one hire. Messages from one hire continue one conversation there,
so its agents stay hired between messages.

```bash
uv run hashiru serve port=9201 budget_usd=0.10 memory_db=data/team_b.sqlite3   # team B
# uncomment team_b under a2a_peers in examples/hashiru.yaml, then:
uv run hashiru serve --config examples/hashiru.yaml                            # team A, on 9200
```

A hub's card declares its whole `budget_usd` as the price of one message, because one message can cost that
much. Team A charges that price for every message, whatever team B really spent (the declared price is a floor),
so a hire of team B needs `budget_usd` per message it will send: here, $0.10 a message, $0.30 for three. The
charges also count toward team A's `max_usd`. If team B has an `api_key`, add it to `team_b` as
`headers: {Authorization: "Bearer ..."}` (through `${oc.env:...}`, not in the file).
