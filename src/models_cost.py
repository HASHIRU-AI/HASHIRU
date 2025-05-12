from dataclasses import dataclass
from typing import Dict
from manager.utils.runtime_selector import detect_runtime_environment

@dataclass
class ModelInfo:
    name: str
    size: float  
    tokens_sec: int
    type: str  
    description: str
    create_cost: int = 0
    invoke_cost: int = 0

class ModelRegistry:
    def __init__(self):
        self.env = detect_runtime_environment()
        self.models = self._build_model_registry()

    def estimate_create_cost(self, size: float, is_api: bool) -> int:
        return int(size * (10 if is_api else 5))

    def estimate_invoke_cost(self, tokens_sec: int, is_api: bool) -> int:
        base_cost = 40 if is_api else 20
        return base_cost + max(0, 60 - tokens_sec)

    def _build_model_registry(self) -> Dict[str, ModelInfo]:
        raw_models = {
            "llama3.2": {
                "size": 3,
                "tokens_sec": 30,
                "type": "local",
                "description": "3B lightweight local model"
            },
            "mistral": {
                "size": 7,
                "tokens_sec": 50,
                "type": "local",
                "description": "7B stronger local model"
            },
            "gemini-2.0-flash": {
                "size": 6,
                "tokens_sec": 170,
                "type": "api",
                "description": "Fast and efficient API model"
            },
            "gemini-2.5-pro-preview-03-25": {
                "size": 10,
                "tokens_sec": 148,
                "type": "api",
                "description": "High-reasoning API model"
            },
            "gemini-1.5-flash": {
                "size": 7,
                "tokens_sec": 190,
                "type": "api",
                "description": "Fast general-purpose model"
            },
            "gemini-2.0-flash-lite": {
                "size": 5,
                "tokens_sec": 208,
                "type": "api",
                "description": "Low-latency, cost-efficient API model"
            },
            "gemini-2.0-flash-live-001": {
                "size": 9,
                "tokens_sec": 190,
                "type": "api",
                "description": "Voice/video low-latency API model"
            }
        }

        
        models = {}
        for name, model in raw_models.items():
            is_api = model["type"] == "api"

            if is_api:
                # Flat cost for all API models
                create_cost, invoke_cost = 20, 50
            else:
                create_cost = self.estimate_create_cost(model["size"], is_api=False)
                invoke_cost = self.estimate_invoke_cost(model["tokens_sec"], is_api=False)

            models[name] = ModelInfo(
                name=name,
                size=model["size"],
                tokens_sec=model["tokens_sec"],
                type=model["type"],
                description=model["description"],
                create_cost=create_cost,
                invoke_cost=invoke_cost
            )
        return models

    def get_filtered_models(self) -> Dict[str, ModelInfo]:
        """Return only models that match the current runtime."""
        if self.env in ["gpu", "cpu-local"]:
            return {k: v for k, v in self.models.items() if v.type == "local"}
        else:
            return {k: v for k, v in self.models.items() if v.type == "api"}

    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Return all models regardless of runtime."""
        return self.models


if __name__ == "__main__":
    registry = ModelRegistry()
    print(f"[INFO] Detected runtime: {registry.env}\n")

    print("Filtered models based on environment:")
    for name, model in registry.get_filtered_models().items():
        print(f"{name}: create={model.create_cost}, invoke={model.invoke_cost}, type={model.type}")
