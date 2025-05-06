from src.utils.runtime_selector import detect_runtime_environment
import os
from dotenv import load_dotenv
load_dotenv()

#Placeholders
MODEL_MAP = {
    "gpu": "gemini-2.5-pro-exp-03-25",
    "cpu-local": "gemini-2.0-flash",     
    "cloud-only": "gemini-2.0-flash"
}
def choose_best_model():
    runtime_env = detect_runtime_environment()
    print(f"[DEBUG] Runtime env: {runtime_env}")
    print(f"[DEBUG] API key exists: {'Yes' if os.environ.get('GEMINI_KEY') else 'No'}")
    if runtime_env == "cpu-local":
        if os.environ.get("GEMINI_KEY"):
            return "gemini-2.0-flash"
        else:
            print("[WARN] No GEMINI_KEY set, falling back to llama3.2.")
            return "llama3.2"