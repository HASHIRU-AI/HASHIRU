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
            "description": "3 Billion parameter model",
            "create_resource_cost": 10,
            "invoke_resource_cost": 10,
        },
        "mistral": {
            "description": "7 Billion parameter model",
            "create_resource_cost": 20,
            "invoke_resource_cost": 50,
        },
        "gemini-2.5-flash-preview-04-17": {
            "description": "Adaptive thinking, cost efficiency",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-2.5-pro-preview-03-25": {
            "description": "Enhanced thinking and reasoning, multimodal understanding, advanced coding, and more",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-2.0-flash": {
            "description": "Next generation features, speed, thinking, realtime streaming, and multimodal generation",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-2.0-flash-lite": {
            "description": "Cost efficiency and low latency",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-1.5-flash": {
            "description": "Fast and versatile performance across a diverse variety of tasks",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-1.5-flash-8b": {
            "description": "High volume and lower intelligence tasks",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-1.5-pro": {
            "description": "Complex reasoning tasks requiring more intelligence",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
        },
        "gemini-2.0-flash-live-001": {
            "description": "Low-latency bidirectional voice and video interactions",
            "create_expense_cost": 20,
            "invoke_expense_cost": 50
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
