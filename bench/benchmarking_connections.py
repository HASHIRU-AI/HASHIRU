from gradio_client import Client
from datasets import load_dataset
import json
import time
import random
import os
from datetime import datetime
import re

def get_last_assistant_content(resp):
    """
    Return the last assistant utterance from the response object
    produced by `client.predict`.
    """
    # ❶ If the server wraps things in a (messages, meta) tuple
    if isinstance(resp, tuple):
        resp = resp[0]

    # ❷ At this point `resp` must be the list of message dicts
    if not isinstance(resp, list):
        return ""

    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue

        # a) plain messages
        if turn.get("content"):
            return turn["content"]

        # b) tool / function_response wrapper
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return out

        # c) messages stored as Part objects inside `content`
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]

    return ""

def benchmark_nyt_connections(num_samples=20, categories=None):
    """
    Benchmark agent performance on NYT connections dataset
    Args:
        num_samples: Number of samples to test
        categories: List of categories to include (None for all)
    """
    # Load NYT connections dataset
    print("Loading NYT connections dataset...")
    dataset = load_dataset("tm21cy/NYT-Connections")
    
    # Initialize client
    client = Client("http://127.0.0.1:7860/")

    # Prepare output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"nyt_connections_benchmark_{timestamp}.jsonl")
    print(f"Results will be saved to {out_path}")
    results = []
    num_samples = min(num_samples, len(dataset["train"])) if num_samples else len(dataset["train"])
    print(f"Sampling {num_samples} samples from the dataset.")
    indices = random.sample(range(len(dataset["train"])), num_samples)
    for i in indices:
        sample = dataset["train"][i]
        if categories and sample["category"] not in categories:
            continue
        print(f"Sample {i}: {sample['contest']}")
        prompt = f"Given the following words, group them into 4 categories of 4 words each:\n{' '.join(sample['words'])}\n\n"
        start_time = time.time()
        response = client.predict(messages=[{"role": "user", "content": prompt}], api_name="/run")
        end_time = time.time()
        elapsed_time = end_time - start_time
        assistant_content = get_last_assistant_content(response)
        results.append({
            "input": sample["words"],
            "date": sample["contest"],
            "output": assistant_content,
            "expected": sample["answers"],
            "elapsed_time": elapsed_time,
        })

        # Save intermediate results
        with open(out_path, "a") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    print(f"Results saved to {out_path}")
    return results

if __name__ == "__main__":
    benchmark_nyt_connections(num_samples=1)