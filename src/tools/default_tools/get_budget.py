
__all__ = ['GetBudget']

from src.manager.budget_manager import BudgetManager


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
        total_resource_budget = budget_manager.get_total_resource_budget()
        current_resource_usage = budget_manager.get_current_resource_usage()
        current_remaining_resource_budget = budget_manager.get_current_remaining_resource_budget()
        
        total_expense_budget = budget_manager.get_total_expense_budget()
        current_expense = budget_manager.get_current_expense()
        current_remaining_expense_budget = budget_manager.get_total_expense_budget() - budget_manager.get_current_expense()
        return {
            "status": "success",
            "message": "Budget retrieved successfully",
            "output": {
                "total_resource_budget": total_resource_budget,
                "current_resource_usage": current_resource_usage,
                "current_remaining_resource_budget": current_remaining_resource_budget,
                "total_expense_budget": total_expense_budget,
                "current_expense": current_expense,
                "current_remaining_expense_budget": current_remaining_expense_budget,
            }
        }
        