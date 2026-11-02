# PURSE round 2: what is built, and the paid batches

Built on 2026-11-01 (no model calls; fake-LLM tests, mutation-checked):

| Roadmap item | What exists now | Switch |
|---|---|---|
| Lost runs (2 of 12 in round 1) | A working turn is capped to leave a closing answer's money; an agent whose working turns come back empty twice gets one closing turn | always on |
| Request out of the CEO's context | Long sections reach the CEO as heading + 2 lines + note; a message naming a section (`j5`, in the text or the hire's job) carries it whole | `request_by_reference`, `purse run --by-reference` |
| Route on measured success | `hire_agent(kind=...)` goes to the cheapest model at or above `route_min_rate`; PURSE fills `success_rates` from the statistics an informed allocator is told | `success_rates`, `purse run --route-on-rates` |
| V5 recalibration | Wider hidden bins: arithmetic digits -1/0/+2 and 10/35/70% multiplications; aggregation 3x near misses (mid gets near-miss regions); strings high all-but-one positional, low without Caesar | `purse pool --wide` (default: round 1's jobs exactly) |
| Third tier | `gpt-5-nano` ($0.05 / $0.40 per M). Set aside: gpt-4.1-nano (deprecated: calls fail), Muse Glimmer 30B on NIM (dearer per token than Luna, and NIM's trial rate limits hit 11 parallel calibrations) and Gemma 4 31B on OpenRouter (no credits) | always in the market once calibrated |
| Unreliable outside agent | `a2a/gridfast`: grids at a quarter of the price, wrong on 40% (deterministic per puzzle) | `csp_agent --name gridfast --wrong 0.4` |

The router orders cloud models by summed per-token price: gpt-5-nano (0.45), gpt-6-luna (0.6), muse-spark-1.3 (5.5).

## Paid batches (each needs approval; estimates from round 1's $0.0015 a job on Luna, $0.0265 on Spark)

0. Done: one Glimmer call on NIM answered and was charged; Glimmer was then replaced by gpt-5-nano.
1. Wide-bin pilot, Luna only: `pool --wide`, then levels 1, 3, 5, two jobs per cell (72 jobs), about $0.15. Go on only
   if Luna's success now differs by bin within a level.
2. Full calibration of the wide pool: four jobs per cell (240 jobs), three attempts each, on Luna, Spark, gpt-5-nano, calc,
   csp and gridfast: about $21, almost all of it Spark. Then `purse check`; recalibrate if V5 still fails.
3. Live arms, 20 projects x 3 seeds each (about $0.03 a run, so about $1.80 an arm): HASHIRU as is, `--by-reference`,
   `--route-on-rates`, both, and the LLM split; about $9. Replayed allocators cost nothing.

About $30 in all. Outside agents must run at the price the market uses (`purse run --price P` and each peer's
`--price P`), unlike round 1, where they charged $0.001 live and c_ref in replay.
