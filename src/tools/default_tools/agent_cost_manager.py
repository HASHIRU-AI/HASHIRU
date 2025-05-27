__all__ = ['AgentCostManager']


class AgentCostManager():
    dependencies = []

    inputSchema = {
        "name": "AgentCostManager",
        "description": "Retrieves the cost of creating and invoking an agent. Also includes the strengths of each model. Please make sure to use this before creating an agent.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    }

    costs = {
        "llama3.2": {
            "description": "Avg Accuracy: 49.75%, Latency 0.9s, 63.4% on multi-task understanding, 40.8% on rewriting, 78.6% on reasoning.",
            "create_resource_cost": 50,
            "invoke_resource_cost": 30,
        },
        "mistral": {
            "description": "Avg Accuracy: 51.3%, Latency 9.7s, 51% on LegalBench, 60.1% on multi-task understanding, 69.9% on TriviaQA, 67.9% on reasoning",
            "create_resource_cost": 75,
            "invoke_resource_cost": 40,
        },
        "deepseek-r1": {
            "description": "Avg Accuracy: 77.3%, Latency: 120s, 69.9% on LegalBench, 71.1% on multi-task understanding, 92.2% on Math",
            "create_resource_cost": 28,
            "invoke_resource_cost": 35,
        },
        "gemini-2.5-flash-preview-05-20": {
            "description": "Avg Accuracy: 75.8%, 82.8% on LegalBench, 81.6% on multi-task understanding, 91.6% on Math",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.15,
        },
        "gemini-2.5-pro-exp-03-25": {
            "description": "Avg Accuracy: 64.3%, 83.6% on LegalBench, 84.1% on multi-task understanding, 95.2% on Math, 63.8% on Coding",
            "create_expense_cost": 0,
            "invoke_expense_cost": 1.25,
        },
        "gemini-2.0-flash": {
            "description": "Avg Accuracy: 64.3%, 79.9% on LegalBench, 77.4% on multi-task understanding, 90.9% on Math, 34.5% on Coding",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.10,
        },
        "gemini-2.0-flash-lite": {
            "description": "Avg Accuracy: 64.1%, 71.6% on multi-task understanding, 86.8% on Math, 28.9% on Coding",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.075
        },
        "gemini-1.5-flash": {
            "description": "62.0% on LegalBench, 61.0% on MMLU, 59.0% on MATH",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.075,
        },
        "gemini-1.5-flash-8b": {
            "description": "High volume and lower intelligence tasks",
            "create_expense_cost": 0,
            "invoke_expense_cost": 0.0375,
        },
    }

    def get_costs(self):
        return self.costs

    def run(self, **kwargs):
        return {
            "status": "success",
            "message": "Cost of creating and invoking an agent",
            "output": self.costs,
        }
