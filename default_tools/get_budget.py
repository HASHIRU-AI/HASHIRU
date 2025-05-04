
__all__ = ['GetBudget']

from src.budget_manager import BudgetManager


class GetBudget():
    dependencies = []

    inputSchema = {
        "name": "GetBudget",
        "description": "Retrieves the current budget available.",
        "parameters": {
            "type": "object",
            "properties":{},
            "required": [],
        },
    }


    def run(self, **kwargs):
        budget_manager = BudgetManager()
        total_budget = budget_manager.get_total_budget()
        current_expense = budget_manager.get_current_expense()
        current_remaining_budget = budget_manager.get_current_remaining_budget()
        return {
            "status": "success",
            "message": "Budget retrieved successfully",
            "output": {
                "total_budget": total_budget,
                "current_expense": current_expense,
                "current_remaining_budget": current_remaining_budget
            }
        }
        