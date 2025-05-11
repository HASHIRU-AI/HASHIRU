
__all__ = ['MemoryManager']

import json


class MemoryManager():
    dependencies = []

    inputSchema = {
        "name": "MemoryManager",
        "description": "Updates, retrieves, or deletes the memory of an AI agent. The memory is stored in a JSON file.",
        "parameters": {
            "type": "object",
            "properties":{
                "action": {
                    "type": "string",
                    "enum": ["add_memory", "get_memory", "delete_memory"],
                    "description": "The action to perform: add_memory, get_memory, or delete_memory.",
                },
                "memory": {
                    "type": "string",
                    "description": "The memory to add. Required for 'add_memory' action.",
                },
                "index": {
                    "type": "integer",
                    "description": "The index of the memory to delete. Required for 'delete_memory' action.",
                },
            },
            "required": ["action"],
        },
    }
    
    def get_memories(self):
        # load the memory from src/data/memory.json
        try:
            with open("src/data/memory.json", "r") as f:
                memory_list = json.load(f)
        except FileNotFoundError:
            memory_list = []
        except json.JSONDecodeError:
            memory_list = []
        return memory_list

    def update_memories(self, memories):
        # save the memory to src/data/memory.json
        with open("src/data/memory.json", "w") as f:
            json.dump(memories, f)


    def run(self, **kwargs):
        # save it to src/data/memory.json
        action = kwargs.get("action")
        memory = kwargs.get("memory")
        index = kwargs.get("index")
        memories = self.get_memories()
        if action == "get_memory":
            return {
                "status": "success",
                "message": "Memory retrieved successfully",
                "output": memories
            }
        if action == "add_memory":
            if memory is None:
                return {
                    "status": "error",
                    "message": "Memory is required for add_memory action",
                    "output": None
                }
            memories.append(memory)
            self.update_memories(memories)
            return {
                "status": "success",
                "message": "Memory created successfully",
                "output": None
            }
        if action == "delete_memory":
            if index is None:
                return {
                    "status": "error",
                    "message": "Index is required for delete_memory action",
                    "output": None
                }
            if index < 0 or index >= len(memories):
                return {
                    "status": "error",
                    "message": "Index out of range",
                    "output": None
                }
            memories.pop(index)
            self.update_memories(memories)
            return {
                "status": "success",
                "message": "Memory deleted successfully",
                "output": None
            }