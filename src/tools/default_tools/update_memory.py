
__all__ = ['UpdateMemory']

import json


class UpdateMemory():
    dependencies = []

    inputSchema = {
        "name": "UpdateMemory",
        "description": "Updates the memory of the AI agent. This tool is used to update the memory of the AI agent with new information.",
        "parameters": {
            "type": "object",
            "properties":{
                "memory": {
                    "type": "string",
                    "description": "The new memory to be added to the AI agent's memory.",
                }
            },
            "required": ["memory"],
        },
    }


    def run(self, **kwargs):
        # save it to src/data/memory.json
        memory = kwargs.get("memory")
        if not memory:
            return {
                "status": "error",
                "message": "Memory is required",
                "output": None
            }
        # add the memory to the memory.json file which is list of strings
        # create the file if it does not exist
        try:
            with open("src/data/memory.json", "r") as f:
                memory_list = json.load(f)
        except FileNotFoundError:
            memory_list = []
        except json.JSONDecodeError:
            memory_list = []
        
        # add the new memory to the list
        memory_list.append(memory)
        # save the list to the file
        with open("src/data/memory.json", "w") as f:
            json.dump(memory_list, f)
        # return the list of memories
        return {
            "status": "success",
            "message": "Memory updated successfully",
            "output": memory_list
        }