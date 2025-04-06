import importlib
import importlib.util
import os
import types
import pip
from google.genai import types

toolsImported = []

TOOLS_DIRECTORY = os.path.abspath("./tools")

class Tool:
    def __init__(self, toolClass):
        self.tool = toolClass()
        self.inputSchema = self.tool.inputSchema
        self.name = self.inputSchema["name"]
        self.description = self.inputSchema["description"]
        self.dependencies = self.tool.dependencies
        for package in self.tool.dependencies:
            try:
                __import__(package.split('==')[0])
            except ImportError:
                print(f"Installing {package}")
                if '==' in package:
                    package = package.split('==')[0]
                pip.main(['install', package])

    def run(self, query):
        return self.tool.run(**query)

class ToolLoader:
    def __init__(self):
        self.toolsImported = []
        self.load_tools()
        pass

    def load_tools(self):
        for filename in os.listdir(TOOLS_DIRECTORY):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, f"{TOOLS_DIRECTORY}/{filename}")
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)
                class_name = foo.__all__[0]
                toolClass = getattr(foo, class_name)
                toolObj = Tool(toolClass)
                self.toolsImported.append(toolObj)

    def runTool(self, toolName, query):
        for tool in self.toolsImported:
            if tool.name == toolName:
                return tool.run(query)
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
            # toolsList.append({
            #     "type": "function",
            #     "function": tool.inputSchema
            # })
        return toolsList

toolLoader = ToolLoader()

# Example usage
# print(toolLoader.getTools())
# print(toolLoader.runTool("AgentCreator", {"agent_name": "Kunla","base_model":"llama3.2","system_prompt": "You love making the indian dish called Kulcha. You declare that in every conversation you have in a witty way." }))