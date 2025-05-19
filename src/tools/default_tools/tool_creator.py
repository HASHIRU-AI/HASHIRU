
__all__ = ['ToolCreator']


class ToolCreator():
    dependencies = []

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
                "tool_code": {
                    "type": "string",
                    "description": "The code of the tool to create",
                },
            },
            "required": ["name", "tool_code"],
        }
    }
    
    def validate_tool_code(self, tool_code):
        # Basic validation to check if the code is a valid Python function
        try:
            compile(tool_code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            print(f"Syntax error in tool code: {e}")
            return False, e

    def run(self, **kwargs):
        print("Running Tool Creator")
        name = kwargs.get("name")
        content = kwargs.get("tool_code")
        print(f"Tool Name: {name}")
        print(f"Tool Content: {content}")
        # Create the tool file
        tool_file_path = f"src/tools/user_tools/{name}.py"
        try:
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
        except Exception as e:
            print(f"Error creating tool: {e}")
            return {
                "status": "error",
                "message": f"Error creating tool: {str(e)}",
                "output": {
                    "tool_file_path": tool_file_path,
                    "tool_name": name,
                }
            }