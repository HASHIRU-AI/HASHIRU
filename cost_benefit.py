import argparse
import subprocess
import time
import requests

# Model info with both ollama and API usage
MODEL_INFO = {
    "mistral": {
        "size": 7,
        "token_cost": 0.002,
        "use_api": False
    },
    "llama": {
        "size": 13,
        "token_cost": 0.0025,
        "use_api": False
    },
    "deepseek": {
        "size": 1.3,
        "token_cost": 0.0015,
        "use_api": False,
        "api": ".."  # Example API
    },
    "gemini": {
        "size": 15,
        "token_cost": 0.003,
        "use_api": False,
        "api": ".."
    }
}

def run_model_ollama(model, prompt):
    try:
        start = time.time()
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        end = time.time()
    except Exception:
        return None

    output = result.stdout.decode().strip()
    duration = end - start
    token_count = len(output.split())
    tokens_per_sec = token_count / duration if duration > 0 else 0
    latency_ms = duration * 1000
    token_cost = MODEL_INFO[model]["token_cost"] * token_count

    return {
        "model": model,
        "latency_ms": latency_ms,
        "tokens_sec": tokens_per_sec,
        "token_cost": token_cost,
        "output": output
    }

def run_model_api(model, prompt):
    try:
        start = time.time()
        response = requests.post(
            MODEL_INFO[model]["api"],
            json={"prompt": prompt},
            timeout=60
        )
        end = time.time()
        response.raise_for_status()
        output = response.json().get("response", "")  # Adjust key as needed
    except Exception:
        return None

    duration = end - start
    token_count = len(output.split())
    tokens_per_sec = token_count / duration if duration > 0 else 0
    latency_ms = duration * 1000
    token_cost = MODEL_INFO[model]["token_cost"] * token_count

    return {
        "model": model,
        "latency_ms": latency_ms,
        "tokens_sec": tokens_per_sec,
        "token_cost": token_cost,
        "output": output
    }

def run_model(model, prompt):
    if MODEL_INFO[model].get("use_api", False):
        return run_model_api(model, prompt)
    else:
        return run_model_ollama(model, prompt)

def get_best_model(prompt, weights, models=None):
    if models is None:
        models = list(MODEL_INFO.keys())

    results = []
    for model in models:
        res = run_model(model, prompt)
        if not res:
            continue

        size = MODEL_INFO[model]["size"]
        cost_score = 1 + weights["w_lat"] * res["latency_ms"] + \
                     weights["w_size"] * size + weights["w_token_cost"] * res["token_cost"]
        benefit_score = weights["w_speed"] * res["tokens_sec"]
        decision_score = benefit_score / cost_score

        res["decision_score"] = decision_score
        results.append(res)

    if not results:
        return "No models succeeded."

    return max(results, key=lambda x: x["decision_score"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose best model for a task")
    parser.add_argument('--prompt', required=True, help='The task or question to ask the models')
    parser.add_argument('--latency', type=int, default=3, help='Priority for latency (1–5)')
    parser.add_argument('--size', type=int, default=3, help='Priority for model size (1–5)')
    parser.add_argument('--cost', type=int, default=3, help='Priority for token cost (1–5)')
    parser.add_argument('--speed', type=int, default=3, help='Priority for tokens/sec (1–5)')
    args = parser.parse_args()

    weights = {
        "w_lat": 0.002 * args.latency,
        "w_size": 0.1 * args.size,
        "w_token_cost": 100 * args.cost,
        "w_speed": 0.01 * args.speed
    }

    best = get_best_model(args.prompt, weights)

    if isinstance(best, str):
        print(best)
    else:
        print(f"\nBest Model: {best['model']}")
        print(f"Decision Score: {round(best['decision_score'], 4)}")
        print(f"Latency (ms): {round(best['latency_ms'], 2)}")
        print(f"Tokens/sec: {round(best['tokens_sec'], 2)}")
        print(f"Token Cost ($): {round(best['token_cost'], 5)}")
        print(f"\nOutput:\n{best['output']}")
