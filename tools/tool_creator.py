import importlib

__all__ = ['ToolCreator']


class ToolCreator():
    dependencies = ["os"]

    inputSchema = {
        "name": "ToolCreator",
        "description": "Creates a tool for the given function",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the tool to create",
                },
                "content": {
                    "type": "string",
                    "description": "The content of the tool to create",
                    "examples": ["""
import importlib

__all__ = ['WeatherApi']


class WeatherApi():
dependencies = ["requests==2.32.3"]

inputSchema = {
    "name": "WeatherApi",
    "description": "Returns weather information for a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for which to get the weather information",
            },
        },
        "required": ["location"],
    }
}

def __init__(self):
    pass

def run(self, **kwargs):
    print("Running Weather API test tool")
    location = kwargs.get("location")
    print(f"Location: {location}")

    requests = importlib.import_module("requests")

    response = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid=ea50e63a3bea67adaf50fbecbe5b3c1e")
    if response.status_code == 200:
        return {
            "status": "success",
            "message": "Weather API test tool executed successfully",
            "error": None,
            "output": response.json()
        }
    else:
        return {
            "status": "error",
            "message": "Weather API test tool failed",
            "error": response.text,
            "output": None
        }

                    """]
                },
            },
            "required": ["name", "content"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        print("Running Tool Creator")
        name = kwargs.get("name")
        content = kwargs.get("content")
        print(f"Tool Name: {name}")
        print(f"Tool Content: {content}")
        # Create the tool file
        tool_file_path = f"tools/{name}.py"
        with open(tool_file_path, "w") as tool_file:
            tool_file.write(content)
        print(f"Tool file created at {tool_file_path}")
        return {
            "status": "success",
            "message": "Tool created successfully",
            "output": {
                "tool_file_path": tool_file_path,
                "tool_name": name,
            }
        }