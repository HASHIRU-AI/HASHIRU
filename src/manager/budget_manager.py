from src.manager.utils.singleton import singleton
import torch
import psutil

@singleton
class BudgetManager():
    total_resource_budget = 100
    current_resource_usage = 0
    total_expense_budget = 10
    current_expense = 0
    is_budget_initialized = False
    is_resource_budget_enabled = True
    is_expense_budget_enabled = True
    
    def __init__(self):
        if not self.is_budget_initialized:
            self.total_resource_budget = self.calculate_total_budget()
            self.is_budget_initialized = True
    
    def set_resource_budget_status(self, status: bool):
        self.is_enabled = status
        if status:
            print("Budget manager is enabled.")
        else:
            print("Budget manager is disabled.")
    
    def set_expense_budget_status(self, status: bool):
        self.is_expense_budget_enabled = status
        if status:
            print("Expense budget manager is enabled.")
        else:
            print("Expense budget manager is disabled.")
        
    def calculate_total_budget(self)-> int:
        total_mem = 0
        gpu_mem = 0
        ram_mem = 0
        if torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            total_vram = torch.cuda.get_device_properties(gpu_index).total_memory
            gpu_mem = total_vram /1024 ** 3
            print(f"GPU detected: {gpu_name}")
            print(f"Total VRAM: {gpu_mem:.2f} GB")
        mem = psutil.virtual_memory()
        ram_mem = mem.total/ 1024 ** 3
        print("No GPU detected. Using CPU.")
        print(f"Total RAM: {ram_mem:.2f} GB")
        total_mem = gpu_mem + ram_mem
        return round((total_mem / 16) * 100)

    def get_total_resource_budget(self):
        return self.total_resource_budget
    
    def get_current_resource_usage(self):
        return self.current_resource_usage
    
    def get_current_remaining_resource_budget(self):
        return self.total_resource_budget - self.current_resource_usage
    
    def can_spend_resource(self, cost):
        if not self.is_resource_budget_enabled:
            return True
        return True if self.current_resource_usage + cost <= self.total_resource_budget else False
        
    def add_to_resource_budget(self, cost):
        if not self.is_resource_budget_enabled:
            return
        if not self.can_spend_resource(cost):
            raise Exception("No resource budget remaining")
        self.current_resource_usage += cost
    
    def remove_from_resource_expense(self, cost):
        if not self.is_resource_budget_enabled:
            return
        if self.current_resource_usage - cost < 0:
            raise Exception("Not enough resource budget to remove")
        self.current_resource_usage -= cost
    
    def get_total_expense_budget(self):
        return self.total_expense_budget
    
    def get_current_expense(self):
        return self.current_expense
    
    def get_current_remaining_expense_budget(self):
        return self.total_expense_budget - self.current_expense
    
    def can_spend_expense(self, cost):
        if not self.is_expense_budget_enabled:
            return True
        return True if self.current_expense + cost <= self.total_expense_budget else False
    
    def add_to_expense_budget(self, cost):
        if not self.is_expense_budget_enabled:
            return
        if not self.can_spend_expense(cost):
            raise Exception("No expense budget remaining")
        self.current_expense += cost