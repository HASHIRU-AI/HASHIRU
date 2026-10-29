# HASHIRU cost extension for A2A

**URI:** `https://github.com/HASHIRU-AI/HASHIRU/blob/main/docs/a2a-cost-extension.md`
**Version:** 1 · **Required:** no

Budget-aware agents need to know what a message will cost before they send it, and what it did cost after. A2A
has no standard way to say either. This extension adds both: an Agent Card declares its prices, and a reply
reports what the request cost.

## Declaring prices (Agent Card)

Add the extension to `capabilities.extensions`:

```json
{
  "uri": "https://github.com/HASHIRU-AI/HASHIRU/blob/main/docs/a2a-cost-extension.md",
  "description": "What a message to this agent costs (USD).",
  "required": false,
  "params": {
    "currency": "USD",
    "per_message_usd": 0.01,
    "per_input_token_usd": 0.0,
    "per_output_token_usd": 0.0
  }
}
```

| Param | Meaning |
|---|---|
| `currency` | Always `"USD"` in version 1. A client ignores the extension if it is anything else. |
| `per_message_usd` | A flat price per request. |
| `per_input_token_usd`, `per_output_token_usd` | Token prices, for agents whose cost depends on length. |

Missing params mean 0. Negative values are treated as 0.

## Reporting cost (reply metadata)

A reply (the final `Message`, or the completed task's status message) may carry what the request really cost:

```json
{ "metadata": { "hashiru_cost_usd": 0.0042 } }
```

## Deadline (request metadata)

A request may carry the Unix time by which its reply is needed. An agent should stop working after it and reply
that it ran out of time:

```json
{ "metadata": { "hashiru_deadline": 1793281727.5 } }
```

## How HASHIRU uses it

- **As a server.** Every agent a HASHIRU hub hosts publishes this extension, with its model's token prices
  (0 for local models). Each reply reports what that agent spent on the request, and each run stops at the
  request's deadline.
- **As a client.** When a manager hires a remote agent (`a2a/<peer>`), the hub reads the peer's Agent Card and
  uses the larger of the declared `per_message_usd` and a `price_usd_per_message` set in the hub's config. A peer
  with neither cannot be hired. Before each message the hirer's pool must cover that price. Afterwards the larger
  of the price and the reported `hashiru_cost_usd` is charged.
