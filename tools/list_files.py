
__all__ = ['ListFiles']

class ListFiles():
    dependencies = []

    inputSchema = {
        "name": "ListFiles",
        "description": "Lists all files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory to list files from",
                },
            },
            "required": ["directory"],
        }
    }

    def run(self, **kwargs):
        print("Running List Files tool")
        directory = kwargs.get("directory")
        print(f"Directory: {directory}")
        try:
            import os
            files = os.listdir(directory)
            return {
                "status": "success",
                "message": "Files listed successfully",
                "error": None,
                "output": files
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "Failed to list files",
                "error": str(e),
                "output": None
            }
