from gradio_client import Client
import geopandas as gpd
import json
import time
import random
import os
from datetime import datetime
from pathlib import Path


def load_countries(geo_path):
    """
    Load country centroids list from a GeoJSON/Shapefile via GeoPandas.
    Returns a list of country names.
    """
    gdf = gpd.read_file(geo_path)
    # pick a name field
    name_field = next((c for c in ["ADMIN","NAME","NAME_EN","NAME_LONG","SOVEREIGN","COUNTRY"] if c in gdf.columns), None)
    if not name_field:
        # fallback to first non-geometry
        non_geom = [c for c in gdf.columns if c.lower() != "geometry"]
        name_field = non_geom[0]

    return gdf[name_field].dropna().unique().tolist()


def benchmark_globle_api(
    server_url: str = "http://127.0.0.1:7860/",
    geo_path: str = "./tools/util/countries.geojson",
    num_trials: int = 10,
    results_dir: str = "results"
):
    """
    Benchmark a GlobleDistanceTool deployed behind a Gradio API.

    Each trial resets the game, reads the hidden target, then issues random guesses until correct or gave up.
    Results are written to JSONL, one line per trial.
    """
    # prepare client and results
    client = Client(server_url)
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(results_dir) / f"globle_api_benchmark_{timestamp}.jsonl"

    # load country list locally
    country_list = load_countries(geo_path)

    all_trials = []
    for trial in range(1, num_trials + 1):
        # reset game
        try:
            client.predict(
                {"geo_path": geo_path, "guess": "", "reset": True},
                api_name="/run"
            )
        except Exception:
            # reset call may error due to missing guess—but state is reset on server
            pass

        # read target from the shared state file
        state = json.loads(Path.home().joinpath(".Globle_distance_state.json").read_text())
        target = state["target"]

        trial_record = {"trial": trial, "target": target, "guesses": []}
        guess_count = 0

        # loop until terminal
        while True:
            guess = random.choice([c for c in country_list if c != target])
            guess_count += 1

            start = time.time()
            out = client.predict(
                {"geo_path": geo_path, "guess": guess, "reset": False},
                api_name="/run"
            )
            latency = time.time() - start

            # parse output structure
            status = out.get("status")
            msg = out.get("message")
            output = out.get("output", {})
            result = output.get("result")

            trial_record["guesses"].append({
                "guess": guess,
                "time_s": latency,
                "status": status,
                "message": msg,
                "output": output
            })

            if result == "gave_up":
                trial_record["failed"] = True
                break
            if result == "correct":
                trial_record["failed"] = False
                trial_record["guess_count"] = guess_count
                break

        # write JSONL line
        with open(results_file, "a") as f:
            f.write(json.dumps(trial_record) + "\n")
        all_trials.append(trial_record)

    # summary
    latencies = [g["time_s"] for t in all_trials for g in t["guesses"]]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"Completed {num_trials} trials. Avg latency: {avg_latency:.3f}s over {len(latencies)} calls.")
    print(f"Results saved to {results_file}")

    return all_trials


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Benchmark GlobleDistanceTool via Gradio API")
    p.add_argument("--server", type=str, default="http://127.0.0.1:7860/", help="Gradio server URL")
    p.add_argument("--geo", type=str, default="./tools/util/countries.geojson", help="Path to geojson file")
    p.add_argument("--trials", type=int, default=10, help="Number of games to run")
    p.add_argument("--outdir", type=str, default="results", help="Output directory for JSONL results")
    args = p.parse_args()

    benchmark_globle_api(
        server_url=args.server,
        geo_path=args.geo,
        num_trials=args.trials,
        results_dir=args.outdir
    )
