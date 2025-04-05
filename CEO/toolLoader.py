import importlib
import importlib.util
import os
import pip

toolsImported = []

TOOLS_DIRECTORY = 'D:/Projects/AI/HASHIRU/tools'

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
        return None

    def getTools(self):
        toolsList = []
        for tool in self.toolsImported:
            toolsList.append({
                "type": "function",
                "function": tool.inputSchema
            })
        return toolsList

toolLoader = ToolLoader()

# Example usage
# print(toolLoader.getTools())
# print(toolLoader.runTool("WeatherApi", {"location": "Davis"}))