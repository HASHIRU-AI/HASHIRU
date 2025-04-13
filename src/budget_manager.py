from src.singleton import singleton

@singleton
class BudgetManager():
    TOTAL_BUDGET = 100
    current_expense = 0
        
    def get_total_budget(self):
        return self.TOTAL_BUDGET
    
    def get_current_expense(self):
        return self.current_expense
    
    def get_current_remaining_budget(self):
        return self.TOTAL_BUDGET - self.current_expense
    
    def can_spend(self, cost):
        return True if self.current_expense + cost <= self.TOTAL_BUDGET else False
        
    def add_to_expense(self, cost):
        if not self.can_spend(cost):
            raise Exception("No budget remaining")
        self.current_expense += cost