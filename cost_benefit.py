import argparse
import subprocess
import time
import requests


def get_best_model(weights: dict, runtime_env: str) -> dict:
    #placeholders
    models = {
        "llama3.2": {"size": 2.5, "token_cost": 0.0001, "speed": 30},
        "mistral": {"size": 4.2, "token_cost": 0.0002, "speed": 50},
        "gemini-2.0-flash": {"size": 6.1, "token_cost": 0.0005, "speed": 60},
        "gemini-2.5-pro-preview-03-25": {"size": 8.2, "token_cost": 0.002, "speed": 45}
    }

    penalty = {
        "gpu": 1.0,
        "cpu-local": 2.0,
        "cloud-only": 1.5
    }

    best_model = None
    best_score = float("-inf")  # Track max score

    for model, metrics in models.items():
        p = penalty.get(runtime_env, 2.0)

        cost_score = (
            weights["w_size"] * metrics["size"] * p +
            weights["w_token_cost"] * metrics["token_cost"] * p +
            weights["w_speed"] * (100 - metrics["speed"])
        )
        benefit_score = weights["w_speed"] * metrics["speed"]

        decision_score = benefit_score / cost_score if cost_score != 0 else 0

        if decision_score > best_score:
            best_score = decision_score
            best_model = model

    if not best_model:
        return "No suitable model found"

    return {
        "model": best_model,
        "score": best_score,
        "token_cost": models[best_model]["token_cost"],
        "tokens_sec": models[best_model]["speed"],
        "output": f"Sample output from {best_model}"
    }
