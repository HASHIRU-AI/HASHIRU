import argparse
import subprocess
import time
import requests

def detect_available_budget(runtime_env: str) -> int:
    import torch
    if "local" in runtime_env and torch.cuda.is_available():
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return min(total_vram_mb, 100)
    else:
        return 100


def get_best_model(runtime_env: str, use_local_only=False, use_api_only=False) -> dict:
    # Model info (cost, tokens/sec, type)
    static_costs = {
        "llama3.2": {"size": 20, "token_cost": 0.0001, "tokens_sec": 30, "type": "local"},
        "mistral": {"size": 40, "token_cost": 0.0002, "tokens_sec": 50, "type": "local"},
        "gemini-2.0-flash": {"size": 60, "token_cost": 0.0005, "tokens_sec": 60, "type": "api"},
        "gemini-2.5-pro-preview-03-25": {"size": 80, "token_cost": 0.002, "tokens_sec": 45, "type": "api"}
    }

    def detect_available_budget(runtime_env: str) -> int:
        import torch
        if "local" in runtime_env and torch.cuda.is_available():
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            return min(total_vram_mb, 100)
        else:
            return 100

    budget = detect_available_budget(runtime_env)

    best_model = None
    best_speed = -1

    for model, info in static_costs.items():
        if info["size"] > budget:
            continue
        if use_local_only and info["type"] != "local":
            continue
        if use_api_only and info["type"] != "api":
            continue
        if info["tokens_sec"] > best_speed:
            best_model = model
            best_speed = info["tokens_sec"]

    if not best_model:
        return {
            "model": "llama3.2",
            "token_cost": static_costs["llama3.2"]["token_cost"],
            "tokens_sec": static_costs["llama3.2"]["tokens_sec"],
            "note": "Defaulted due to no models fitting filters"
        }

    return {
        "model": best_model,
        "token_cost": static_costs[best_model]["token_cost"],
        "tokens_sec": static_costs[best_model]["tokens_sec"]
    }
