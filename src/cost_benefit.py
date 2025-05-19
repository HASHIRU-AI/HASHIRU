import argparse
import subprocess
import time
import requests

def detect_available_budget(runtime_env: str) -> int:
    """
    Return an approximate VRAM‑based budget (MB) when running locally,
    else default to 100.
    """
    import torch
    if "local" in runtime_env and torch.cuda.is_available():
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return min(total_vram_mb, 100)
    return 100

def get_best_model(runtime_env: str, *, use_local_only: bool = False, use_api_only: bool = False) -> dict:
    """
    Pick the fastest model that fits in the detected budget while
    respecting the locality filters.
    """
    static_costs = {
        "llama3.2":  {"size": 20, "token_cost": 0.0001, "tokens_sec": 30, "type": "local"},
        "mistral":   {"size": 40, "token_cost": 0.0002, "tokens_sec": 50, "type": "local"},
        "gemini-2.0-flash":            {"size": 60, "token_cost": 0.0005, "tokens_sec": 60, "type": "api"},
        "gemini-2.5-pro-preview-03-25": {"size": 80, "token_cost": 0.002 , "tokens_sec": 45, "type": "api"},
    }

    budget = detect_available_budget(runtime_env)
    best_model, best_speed = None, -1

    for model, info in static_costs.items():
        if info["size"] > budget:
            continue
        if use_local_only and info["type"] != "local":
            continue
        if use_api_only and info["type"] != "api":
            continue
        if info["tokens_sec"] > best_speed:
            best_model, best_speed = model, info["tokens_sec"]

    chosen = best_model or "llama3.2"  # sensible default
    return {
        "model": chosen,
        "token_cost": static_costs[chosen]["token_cost"],
        "tokens_sec": static_costs[chosen]["tokens_sec"],
        "note": None if best_model else "Defaulted because no model met the constraints",
    }
