__all__ = ['AgentCostManager']


class AgentCostManager():
    dependencies = []

    inputSchema = {
        "name": "AgentCostManager",
        "description": "Retrieves the cost of creating and invoking an agent. Please make sure to use this before creating an agent.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    }

    costs = {
        "llama3.2": {
            "description": "The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.",
            "create_resource_cost": 50,
            "invoke_resource_cost": 30,
        },
        "mistral": {
            "description": "One of the most powerful open source models for its size. It is vastly superior in code and reasoning benchmarks.",
            "create_resource_cost": 75,
            "invoke_resource_cost": 40,
        },
        "deepseek-r1": {
            "description": "DeepSeek's first-generation reasoning models, achieving performance comparable to OpenAI-o1 across math, code, and reasoning tasks.",
            "create_resource_cost": 28,
            "invoke_resource_cost": 35,
        },
        "gemini-2.5-flash-preview-04-17": {
            "description": "Adaptive thinking, cost efficiency",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.15,
        },
        "gemini-2.5-pro-preview-05-06": {
            "description": "Enhanced thinking and reasoning, multimodal understanding, advanced coding, and more",
            "create_expense_cost": 0,
            "invoke_expense_cost": 1.25,
        },
        "gemini-2.0-flash": {
            "description": "Next generation features, speed.",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.10,
        },
        "gemini-2.0-flash-lite": {
            "description": "Cost efficiency and low latency",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.075
        },
        "gemini-1.5-flash": {
            "description": "Fast and versatile performance across a diverse variety of tasks",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.075,
        },
        "gemini-1.5-flash-8b": {
            "description": "High volume and lower intelligence tasks",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.0375,
        },
        "gemini-1.5-pro": {
            "description": "Complex reasoning tasks requiring more intelligence",
            "create_expense_cost": 0,
            "invoke_expense_cost": 1.25,
        },
        "gemini-2.0-flash-live-001": {
            "description": "Low-latency bidirectional voice and video interactions",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.50,
        }
    }

    def get_costs(self):
        return self.costs

    def run(self, **kwargs):
        return {
            "status": "success",
            "message": "Cost of creating and invoking an agent",
            "output": self.costs,
        }
