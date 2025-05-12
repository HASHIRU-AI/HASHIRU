from src.manager.utils.singleton import singleton
import torch
import psutil

@singleton
class BudgetManager():
    TOTAL_BUDGET = 100
    current_expense = 0
    is_budget_initialized = False
    def __init__(self):
        if not self.is_budget_initialized:
            self.TOTAL_BUDGET = self.set_total_budget()
            self.is_budget_initialized = True
        
    def set_total_budget(self)-> int:
        total_mem = 0
        if torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            total_vram = torch.cuda.get_device_properties(gpu_index).total_memory
            total_mem = total_vram /1024 ** 3
            print(f"GPU detected: {gpu_name}")
            print(f"Total VRAM: {total_mem:.2f} GB")
        else:
            mem = psutil.virtual_memory()
            total_mem = mem.total/ 1024 ** 3
            print("No GPU detected. Using CPU.")
            print(f"Total RAM: {total_mem:.2f} GB")
        return round((total_mem / 16) * 100)

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
    
    def remove_from_expense(self, cost):
        if self.current_expense - cost < 0:
            raise Exception("Cannot remove more than current expense")
        self.current_expense -= cost