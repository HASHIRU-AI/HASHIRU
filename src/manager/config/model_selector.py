from src.manager.utils.runtime_selector import detect_runtime_environment
from src.cost_benefit import get_best_model
import os
from dotenv import load_dotenv
load_dotenv()

def choose_best_model(return_full=False):
    env = detect_runtime_environment()
    print(f"[INFO] Runtime Environment: {env}")

    result = get_best_model(env)

    if not result.get("model"):
        print("[WARN] No model found under budget — using fallback.")
        fallback_model = "gemini-2.0-flash" if os.getenv("GEMINI_KEY") else "llama3.2"
        return {"model": fallback_model} if return_full else fallback_model

    print(f"[INFO] Auto-selected model: {result['model']} (token cost: {result['token_cost']}, tokens/sec: {result['tokens_sec']})")
    return result if return_full else result["model"]
