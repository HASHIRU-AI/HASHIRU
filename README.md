# HASHIRU: Hierarchical Agent System for Hybrid Intelligent Resource Utilization
(For AgentX competition)

![HASHIRU_ARCH](HASHIRU_ARCH.png)

## Overview
HASHIRU is an agent-based framework designed to dynamically allocate and manage large language models (LLMs) and external APIs through a CEO model. The CEO model acts as a central manager, capable of hiring, firing, and directing multiple specialized agents (employees) over a given budget. It can also create and utilize external APIs as needed, making it highly flexible and scalable.

## High-Level Overview

1. Loads available tools using ToolLoader
2. Instantiates a Gemini-powered CEO using GeminiManager
3. Wraps the user prompt into a structured Content object
4. Calls Gemini with the prompt
5. Executes external tool calls if needed
6. Returns a full response to the user
7. Can ask the user for clarification if needed

After every step in the request, there is a checkpoint so that we can check what happened in that step (i.e., which tool was called, what was the response, etc.). This is useful for debugging and understanding the flow of the program.

## Features
- **Cost-Benefit Matrix**:  
  Selects the best LLM model (LLaMA, Mixtral, Gemini, DeepSeek, etc.) for any task using Ollama, based on latency, size, cost, quality, and speed.
- **Dynamic Agent Management**:  
  The CEO agent dynamically hires and fires specialized agents based on task requirements and budget constraints.
- **API Integration**:  
  Seamlessly integrates external APIs for enhanced functionality and scalability.

## How to Run

Clone the repo and install the required dependencies:

```bash
git clone <repository-url>
cd HASHIRU
pip install -r requirements.txt
```

Run the main script:

```bash
python main.py
```

## Usage

### Cost-Benefit Matrix

The `cost_benefit.py` tool allows you to select the best LLM model for a given task based on customizable weights:

```bash
python tools/cost_benefit.py \
  --prompt "Best places to visit in Davis" \
  --latency 4 --size 2 --cost 5 --speed 3
```

Each weight is on a scale of **1** (least important) to **5** (most important):

- `--latency`: Prefer faster responses (lower time to answer)
- `--size`: Prefer smaller models (use less memory/resources)
- `--cost`: Prefer cheaper responses (fewer tokens, lower token price)
- `--speed`: Prefer models that generate tokens quickly (tokens/sec)

### Example Output

```plaintext
Selected Model: Gemini
Reason: Optimal balance of cost, speed, and latency for the given weights.
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

