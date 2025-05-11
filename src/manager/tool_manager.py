import importlib
import importlib.util
import os
import types
from typing import List
import pip
from google.genai import types
import sys

from src.manager.budget_manager import BudgetManager
from src.manager.utils.singleton import singleton
from src.manager.utils.suppress_outputs import suppress_output
from src.default_tools.get_agents_tool import GetAgents
from src.default_tools.tool_deletor import ToolDeletor
from src.manager.utils.streamlit_interface import output_assistant_response

toolsImported = []

TOOLS_DIRECTORIES = [os.path.abspath("./src/default_tools"), os.path.abspath("./src/tools")]

class Tool:
    def __init__(self, toolClass):
        suppress_output(self.load_tool)(toolClass)
        
    def load_tool(self, toolClass):
        self.tool = toolClass()
        self.inputSchema = self.tool.inputSchema
        self.name = self.inputSchema["name"]
        self.description = self.inputSchema["description"]
        self.dependencies = self.tool.dependencies
        self.create_cost = None
        self.invoke_cost = None
        if "create_cost" in self.tool.inputSchema:
            self.create_cost = self.tool.inputSchema["create_cost"]
        if "invoke_cost" in self.tool.inputSchema:
            self.invoke_cost = self.tool.inputSchema["invoke_cost"]
        if self.dependencies:
            self.install_dependencies()
    
    def install_dependencies(self):
        for package in self.dependencies:
            try:
                __import__(package.split('==')[0])
            except ImportError:
                print(f"Installing {package}")
                if '==' in package:
                    package = package.split('==')[0]
                pip.main(['install', package])

    def run(self, query):
        return self.tool.run(**query)

@singleton
class ToolManager:
    toolsImported: List[Tool] = []
    budget_manager: BudgetManager = BudgetManager()

    def __init__(self):
        self.load_tools()

        output_assistant_response(f"Budget Remaining: {self.budget_manager.get_current_remaining_budget()}")

    def load_tools(self):
        newToolsImported = []
        for directory in TOOLS_DIRECTORIES:
            for filename in os.listdir(directory):
                if filename.endswith(".py") and filename != "__init__.py":
                    module_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(module_name, f"{directory}/{filename}")
                    foo = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(foo)
                    class_name = foo.__all__[0]
                    toolClass = getattr(foo, class_name)
                    toolObj = Tool(toolClass)
                    newToolsImported.append(toolObj)
                    if toolObj.create_cost is not None:
                        self.budget_manager.add_to_expense(toolObj.create_cost)
        self.toolsImported = newToolsImported

    def runTool(self, toolName, query):
        output_assistant_response(f"Budget Remaining: {self.budget_manager.get_current_remaining_budget()}")
        for tool in self.toolsImported:
            if tool.name == toolName:
                if tool.invoke_cost is not None:
                    self.budget_manager.add_to_expense(tool.invoke_cost)
                return tool.run(query)
        output_assistant_response(f"Budget Remaining: {self.budget_manager.get_current_remaining_budget()}")
        return {
            "status": "error",
            "message": f"Tool {toolName} not found",
            "output": None
        }
    

    def getTools(self):
        toolsList = []
        for tool in self.toolsImported:
            parameters = types.Schema()
            parameters.type = tool.inputSchema["parameters"]["type"]
            properties = {}
            for prop, value in tool.inputSchema["parameters"]["properties"].items():
                properties[prop] = types.Schema(
                    type=value["type"],
                    description=value["description"]
                )
            parameters.properties = properties
            parameters.required = tool.inputSchema["parameters"].get("required", [])
            function = types.FunctionDeclaration(
                name=tool.inputSchema["name"],
                description=tool.inputSchema["description"],
                parameters=parameters,
            )
            toolType = types.Tool(function_declarations=[function])
            toolsList.append(toolType)
        return toolsList
    
    def delete_tool(self, toolName, toolFile):
        try:
            tool_deletor = ToolDeletor()
            tool_deletor.run(name=toolName, file_path=toolFile)
            for tool in self.toolsImported:
                if tool.name == toolName:
                    # remove budget for the tool
                    if tool.create_cost is not None:
                        self.budget_manager.remove_from_expense(tool.create_cost)
                    self.toolsImported.remove(tool)
                    return {
                        "status": "success",
                        "message": f"Tool {toolName} deleted",
                        "output": None
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool {toolName} not found",
                "output": None
            }