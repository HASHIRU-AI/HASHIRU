# HASHIRU v2 launch posts (drafts, for Kunal to edit and post)

Site: https://hashiru-ai.github.io/hashiru-website/
PDF: https://hashiru-ai.github.io/hashiru-website/hashiru-v2-whitepaper.pdf
Code: https://github.com/HASHIRU-AI/HASHIRU

Images in this folder: `card.png` (the link preview, 1200x630), `trace.png` (one hire's money, from the site's hero),
`budget-chart.png` (the budget sweep).

## Before posting

- [ ] Push the second round's runtime to `main` on HASHIRU-AI/HASHIRU. The site's `team.yaml` uses
      `request_by_reference` and `success_rates`, which are not on public `main` yet, so a reader who copies it today
      gets an error.
- [ ] Harshil has read the PDF and is happy with it going out under his name.
- [ ] Post on a weekday morning (US Pacific) so the HN and Reddit posts get a full day.

---

## X / Twitter thread

**1/** (attach `card.png`, or let the link preview show it)
We gave a team of LLM agents a budget and let them spend it, hiring models and other agents. HASHIRU v2 is the system,
and the whitepaper measures how they spend (including what still goes wrong).
https://hashiru-ai.github.io/hashiru-website/

**2/**
Each agent gets a budget from the agent that hired it. The runtime caps every call at what the agent has left before
the call is made, refunds what a fired hire did not spend, and makes an agent on its last step answer.

**3/**
The guards hold. Over 551 live hires, none spent more than 1.11 times its budget. The first version of the same agents
let one hire spend 372 times its budget and lost half their runs.

**4/**
Where the agents win: hard jobs where the market sells a specialist. On hard arithmetic and logic grids they were right
79% of the time, against 69% for the cheap model alone. Overall they match it (0.737 vs 0.740), at 3.9x its
spend.

**5/** (attach `trace.png`)
What still goes wrong: they pick the expensive model for the right jobs but give it too little money. Its 79 hires took
55% of the spend and returned 12 right answers. The fix we want first is to quote the price when the hire is made.

**6/** (attach `budget-chart.png`)
In replay, as the budget grows eightfold the cheap model alone stays near 0.75 while a cascade climbs to 0.88. That is
where agents that split the money should win, and where we have not run them live yet.

**7/**
With Harshil Patel (UC Davis). The code is source-available (PolyForm Noncommercial):
https://github.com/HASHIRU-AI/HASHIRU

---

## LinkedIn

A team of language-model agents spends money every time it hires a strong model for a hard step, a cheap one for an
easy step, or an outside agent that sells an exact answer. In HASHIRU v2, Harshil Patel and I gave that money to the
agents themselves, and measured how they spend it.

Each agent gets a budget from the agent that hired it, and spends it hiring models and other agents over A2A and MCP.
The runtime keeps the spending safe. Every call is capped at what the agent has left, unspent money comes back when a
hire is fired, and an agent on its last step must answer.

What we found:
• The guards hold. Over 551 live hires, none spent more than 1.11 times its budget (the first version let one spend 372
  times).
• The agents win where the market has a specialist. On hard arithmetic and logic grids they were right 79% of the time,
  against 69% for the cheap model alone.
• Overall they match the cheap model (0.737 against 0.740), at 3.9 times its spend, because they give the
  expensive model too little money. That is the first thing we want to fix.

These are small runs ($33 of model calls in all), and the whitepaper says so, with every interval and each gap's next
step.

Whitepaper and site: https://hashiru-ai.github.io/hashiru-website/
Code: https://github.com/HASHIRU-AI/HASHIRU

---

## Hacker News (Show HN)

**Title:** Show HN: HASHIRU v2, LLM agent teams that hire and spend within a budget
**URL:** https://hashiru-ai.github.io/hashiru-website/

**First comment (post it right after submitting):**

Hi HN, I'm Kunal, one of the two authors. HASHIRU v2 gives a team of LLM agents a budget: the CEO agent holds a budget
in dollars, and hiring a model or an outside A2A agent sets part of it aside for that hire. The hub (a Python runtime on
LiteLLM and LangGraph) does the accounting. A call is refused before it reaches the provider if the hire cannot pay for
a short answer, output is capped to what the hire holds, firing refunds the rest, and an agent on its last step is made
to answer instead of failing.

What we measured, on a benchmark of priced jobs we built for this:

- Over 551 live hires none spent more than 1.11x its budget. The 11% overrun is our token counter running low on
  long inputs. The first version let one hire spend 372x.
- The agents beat the cheap model alone on hard jobs where a specialist agent exists (79% vs 69% right), but overall
  they only match it (0.737 vs 0.740) at 3.9x the spend.
- The main reason is that they hire the expensive model for the right jobs and then give it too little money: 78 of its
  79 hires had a call refused because their budget could not pay for it.

It's small (about $33 of model calls), the outcome tables fail three of our seven validity checks, and the agents ran
live at one budget only. The PDF has every interval. The code is source-available under PolyForm Noncommercial rather
than OSI open source, which I know matters here. Happy to answer questions about the accounting or the benchmark.

---

## Reddit, r/MachineLearning

**Title:** [R] Agents That Hold the Money: giving LLM agent teams a budget, and measuring how they spend it (HASHIRU v2)

**Body:**

We gave a team of LLM agents a dollar budget and let them hire models and priced outside agents (over A2A and MCP)
with it. Each hire's budget comes out of its parent's, every call is capped at what the hire has left, unspent money comes back
when a hire is fired, and an agent on its last step must answer.

Results (small runs, $33.21 of model calls in all):

- **Guards:** over 551 live hires in 120 runs, the worst overrun was 1.11x the hire's budget (372x before the call cap).
- **Context:** holding long inputs out of the CEO's context raised the score by 0.136 (95% CI [+0.067, +0.216]).
- **Where agents win:** hard jobs with a specialist in the market, 79% vs 69% right for the cheap model alone. Overall
  they match it, 0.737 vs 0.740, at 3.9x the spend.
- **Why not more:** they choose the expensive model for hard jobs but give it too little money (79 hires, 55% of spend,
  12 right answers).
- **Budget sweep (replay only):** the cheap model alone stays near 0.75 as the budget grows 8x while a cascade reaches
  0.88, so a budget with room is where a strategy that splits the money should win. We have not run the agents there yet.

Limitations we report up front: both outcome tables fail three of seven validity checks (so we cannot show what
adapting during the task is worth), the agents ran live at one budget, and the market has three cloud models.

Whitepaper: https://hashiru-ai.github.io/hashiru-website/
Code: https://github.com/HASHIRU-AI/HASHIRU

---

## A2A community (GitHub Discussions on the A2A project, or its Discord)

**Title:** Pricing remote agents in A2A: a small extension and what we measured with it

HASHIRU v2 hires remote A2A agents as priced workers. We added a small extension for it: an agent card declares a price
per message or per token, a reply reports what the request cost, and the hub charges the larger of the two to the
hire's budget. Because a HASHIRU hub is itself an A2A agent, one hub can hire a whole team as one priced worker.

In our runs, a priced logic-grid solver hired this way answered all 41 of its messages right, and hard jobs with a
specialist like it are where the agents beat a cheap model alone (79% against 69%). The whitepaper describes the
extension and the accounting around it: https://hashiru-ai.github.io/hashiru-website/

We would like feedback on whether a price field like this belongs in the card, and how it should sit next to the x402
payments extension.
