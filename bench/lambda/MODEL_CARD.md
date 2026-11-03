---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
base_model_relation: finetune
library_name: transformers
tags:
  - lora
  - peft
  - budget-allocation
  - multi-agent
  - hashiru
  - purse
---

# PURSE splitter (Qwen3.5 9B)

This model splits a money budget across jobs. It reads a project and writes one JSON reply that says which model or
outside agent should do each job, or to skip it. It is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
fine-tuned with LoRA to copy the split that a knapsack solver makes from the same numbers.

It comes from PURSE, the budget benchmark in [HASHIRU](https://github.com/HASHIRU-AI/HASHIRU)
([project site](https://hashiru-ai.github.io/hashiru-website)). It is a research result, not a general assistant.

## What it does

A project gives:

- a budget in dollars. The splitter's own reply is paid from it first;
- the options: three cloud models and three outside agents, with their prices;
- a table of measured success rate and mean cost per attempt, for each job family, difficulty level and option;
- eight jobs, each with its family, level, point value and the first 160 characters of its text.

The model replies with only a JSON object, for example:

```json
{"j1": "skip", "j2": "skip", "j3": "a2a/gridfast", "j4": "skip", "j5": "skip", "j6": "gpt-6-luna", "j7": "gpt-6-luna", "j8": "gpt-6-luna"}
```

Each job then gets one attempt on its option, most valuable first. An attempt the money left cannot pay for is cut off.

## Results

We scored it on 20 projects it never saw in training, each at four budgets. The score is the share of points earned
(each job weighted by its value), averaged over three replays of the stored outcomes. τ is the budget as a multiple of
what the best full run costs.

| Splitter | τ 0.25 | τ 0.5 | τ 1 | τ 2 |
|---|---|---|---|---|
| Knapsack solver (free, reads no job text) | 0.715 | 0.788 | 0.842 | 0.902 |
| **This model** | **0.663** | **0.750** | **0.820** | **0.879** |
| Qwen3.5 9B untrained, thinking off | 0.513 | 0.578 | 0.660 | 0.661 |
| Qwen3.5 9B untrained, thinking on | 0 | 0 | 0 | 0 |
| gpt-6-luna split | | 0.695 | | |

- Training adds 0.15 to 0.22 over the same model untrained. The 95% intervals over the 20 projects exclude zero at
  every budget.
- It still loses to the free solver, by 0.02 to 0.05. It pays for its reply (about 8% of the budget), and it can at
  best copy the solver. It agrees with the solver on 92% of jobs.
- Against gpt-6-luna's split at τ 0.5 it is 0.056 higher, but the interval includes zero (-0.031 to +0.131).
- With thinking on, the untrained model never finished thinking within 8,192 tokens at temperature 0, so it answered
  nothing.

In short: a 9B model can learn the solver's splits. Unless you need a model to read the jobs' text, use the solver.

## How to use

Serve the merged model with vLLM. The checkpoint keeps Qwen3.5's vision tower, so serve the language model only:

```bash
vllm serve guineapig/purse-splitter-qwen3.5-9b --reasoning-parser qwen3 --language-model-only --max-model-len 16384
```

Ask with the project as a single user message, thinking off and temperature 0, as in training:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
reply = client.chat.completions.create(
    model="guineapig/purse-splitter-qwen3.5-9b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
split = reply.choices[0].message.content
```

Or add the adapter in `adapter/` to the base model yourself:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "guineapig/purse-splitter-qwen3.5-9b", subfolder="adapter")
```

The prompt must follow PURSE's format. `split_text` in
[`bench/purse/splitter.py`](https://github.com/HASHIRU-AI/HASHIRU/blob/main/bench/purse/splitter.py) builds it.

## Limits

- It learned one market: these six options, their prices, and four job families (arithmetic, grids, aggregation,
  strings) at levels 1 to 5. With other options, prices or families it will likely need new training.
- Every training project has eight jobs.
- The training targets set aside the reply's cost from a rough estimate (about 1,600 tokens), but real prompts are about
  3,200 tokens. So the targets slightly underestimate what the reply costs.

## Training

- **Data:** 3,000 training and 200 validation prompts from PURSE's second round, from projects that share no job with
  the 20 test projects. Budgets are 0.25, 0.5, 1 and 2 times the full cost. The target is the knapsack's split on the
  budget minus the reply's cost. The scripts that build it are in
  [`bench/lambda/`](https://github.com/HASHIRU-AI/HASHIRU/tree/main/bench/lambda).
- **Method:** LoRA with rank 16, alpha 32 and dropout 0.05, on the language model's attention and MLP projections.
  The loss counts only the reply tokens. The chat template is used with thinking off.
- **Settings:** one epoch (188 steps), 16 examples per step, learning rate 1e-4 with 6 warmup steps, longest example
  4,096 tokens, seed 2026. Built with the Transformers Trainer, not TRL.
- **Loss:** training loss fell from 0.137 to about 0.02. Validation loss ended at 0.020.
- **Hardware:** one NVIDIA H100 80GB, about 33 minutes.
- **Software:** transformers 5.20.2, peft 0.21.2, torch 2.14.0 (CUDA 13.0), flash-linear-attention 0.6.0. Evaluated
  with vLLM 0.33.0. The base model is revision c2022362 of Qwen/Qwen3.5-9B.

## Files

- Root: the merged model (base plus adapter) in bfloat16, about 18.8 GB.
- `adapter/`: the LoRA adapter alone, about 112 MB.
