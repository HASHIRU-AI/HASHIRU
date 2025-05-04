from gradio_client import Client
from datasets import load_dataset
import requests
import json
import time
import random
import os
import re
from datetime import datetime

# Fetch the official Wordle guess list from GitHub
WORD_LIST_URL = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"

def load_word_list():
    resp = requests.get(WORD_LIST_URL)
    resp.raise_for_status()
    words = [w.strip().lower() for w in resp.text.splitlines()]
    return [w for w in words if len(w) == 5 and w.isalpha()]

WORD_LIST = load_word_list()


def get_last_assistant_content(resp):
    if isinstance(resp, tuple): resp = resp[0]
    if not isinstance(resp, list): return ""
    for turn in reversed(resp):
        if turn.get("role") != "assistant": continue
        if turn.get("content"): return turn["content"]
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out: return out
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"): return parts[0]["text"]
    return ""


def compute_feedback(guess, solution):
    feedback = ["B"] * 5
    sol = list(solution)
    for i, g in enumerate(guess):
        if g == sol[i]: feedback[i], sol[i] = "G", None
    for i, g in enumerate(guess):
        if feedback[i] == "B" and g in sol:
            feedback[i] = "Y"
            sol[sol.index(g)] = None
    return "".join(feedback)


def sanitize_guess(raw):
    raw = raw.lower()
    m = re.search(r"\b[a-z]{5}\b", raw)
    if m: return m.group(0)
    cleaned = re.sub(r"[^a-z]", "", raw)
    return cleaned[-5:]


def benchmark_wordle(num_games=10, max_guesses=6):
    client = Client("http://127.0.0.1:7860/")
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"wordle_benchmark_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    results = []

    for gi in range(num_games):
        solution = random.choice(WORD_LIST)
        print(f"Game {gi+1}/{num_games}, solution: {solution}")
        guesses = []
        attempts = 0
        start_time = time.time()

        while attempts < max_guesses:
            history = "\n".join(f"Guess: {g}, Feedback: {f}" for g, f in guesses)
            prompt = (
                f"Wordle game. Guess the 5-letter word.\n" +
                (history + "\n" if history else "") +
                "Respond with a single 5-letter guess and with ONLY YOUR GUESS. NO FILLER OR PUNCTUATION."
                "\n" + "Use the feedback format: G=green, Y=yellow, B=black.\n" +
                f"(Green: letter in correct position, Yellow: letter in wrong position, Black: letter not in word)\n" +
                f"Use tools and agents to help you guess the word.\n"
            )
            resp = client.predict(messages=[{"role": "user", "content": prompt}], api_name="/run")
            raw = get_last_assistant_content(resp)
            guess = sanitize_guess(raw)

            # If guess invalid, retry without counting
            if len(guess) != 5 or guess not in WORD_LIST:
                print(f"Warning: '{guess}' invalid; retrying without using a turn.")
                continue

            feedback = compute_feedback(guess, solution)
            guesses.append((guess, feedback))
            attempts += 1
            print(f"Attempt {attempts}: {guess} -> {feedback}")
            if feedback == "GGGGG":
                break

        results.append({
            "solution": solution,
            "guesses": guesses,
            "solved": bool(guesses and guesses[-1][1] == "GGGGG"),
            "turns": len(guesses),
            "time": time.time() - start_time
        })
        with open(out_path, "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

    print(f"Benchmark complete, results saved to {out_path}")
    return results


if __name__ == "__main__":
    benchmark_wordle(num_games=1)
