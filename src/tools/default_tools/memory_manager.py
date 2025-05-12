
__all__ = ['MemoryManager']

import json


class MemoryManager():
    dependencies = []

    inputSchema = {
        "name": "MemoryManager",
        "description": "Updates, retrieves, or deletes the memory of an AI agent.",
        "parameters": {
            "type": "object",
            "properties":{
                "action": {
                    "type": "string",
                    "enum": ["add_memory", "get_all_memories", "delete_memory"],
                    "description": "The action to perform: add_memory, get_all_memories, or delete_memory.",
                },
                "memory": {
                    "type": "string",
                    "description": "The memory to add. Required for 'add_memory' action.",
                },
                "key": {
                    "type": "string",
                    "description": "The key to delete or add memory."
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
            json.dump(memories, f, indent=4)


    def run(self, **kwargs):
        # save it to src/data/memory.json
        action = kwargs.get("action")
        memory = kwargs.get("memory")
        key = kwargs.get("key")
        memories = self.get_memories()
        if action == "get_all_memories":
            return {
                "status": "success",
                "message": "Memory retrieved successfully",
                "output": memories
            }
        if action == "add_memory":
            if memory is None or key is None:
                return {
                    "status": "error",
                    "message": "Memory and key are required for add_memory action",
                    "output": None
                }
            # check if the key already exists
            for mem in memories:
                if mem["key"] == key:
                    return {
                        "status": "error",
                        "message": f"Memory with key {key} already exists",
                        "output": None
                    }
            memories.append({
                "key": key,
                "memory": memory
            })
            self.update_memories(memories)
            return {
                "status": "success",
                "message": "Memory created successfully",
                "output": None
            }
        if action == "delete_memory":
            if key is None:
                return {
                    "status": "error",
                    "message": "Key is required for delete_memory action",
                    "output": None
                }
            # check if the key exists
            for mem in memories:
                if mem["key"] == key:
                    memories.remove(mem)
                    self.update_memories(memories)
                    return {
                        "status": "success",
                        "message": "Memory deleted successfully",
                        "output": None
                    }
            return {
                "status": "error",
                "message": f"Memory with key {key} not found",
                "output": None
            }