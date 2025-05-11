from src.manager.utils.runtime_selector import detect_runtime_environment
from cost_benefit import get_best_model
import os
from dotenv import load_dotenv
load_dotenv()

def choose_best_model(return_full=False):
    env = detect_runtime_environment()
    print(f"[INFO] Runtime Environment: {env}")
    
    weights = {
        "w_size": 0.1,
        "w_token_cost": 100,
        "w_speed": 0.5
    }

    result = get_best_model(weights, env)

    if isinstance(result, str) or not result.get("model"):
        if env == "cpu-local":
            if os.getenv("GEMINI_KEY"):
                print("[INFO] Falling back to Gemini for cpu-local.")
                return {"model": "gemini-2.0-flash"} if return_full else "gemini-2.0-flash"
            else:
                print("[WARN] GOOGLE_API_KEY missing. Falling back to llama3.2.")
                return {"model": "llama3.2"} if return_full else "llama3.2"
        return {"model": "llama3.2"} if return_full else "llama3.2"

    print(f"[INFO] Auto-selected model: {result['model']}")
    return result if return_full else result["model"]
