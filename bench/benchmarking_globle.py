#!/usr/bin/env python3
import random
import math
import sys
import json
import time
import difflib
import os
import requests
import re
import geopandas as gpd
from shapely.geometry import Point
from gradio_client import Client
from datetime import datetime

# -----------------------------------------------------------------------------
# Utility: haversine distance only
# -----------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# -----------------------------------------------------------------------------
# Load country centroids and geometries
# -----------------------------------------------------------------------------

def load_countries(geo_path):
    gdf = gpd.read_file(geo_path)
    candidates = ["ADMIN","NAME","NAME_EN","NAME_LONG","SOVEREIGN","COUNTRY"]
    name_field = next((f for f in candidates if f in gdf.columns), None)
    if name_field is None:
        non_geom = [c for c in gdf.columns if c.lower()!='geometry']
        name_field = non_geom[0] if non_geom else None
    centroids, geoms = {}, {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        if not geom or geom.is_empty: continue
        c = geom.centroid
        country = row[name_field]
        centroids[country] = (c.y, c.x)
        geoms[country] = geom
    return centroids, geoms

# -----------------------------------------------------------------------------
# Extract last assistant content
# -----------------------------------------------------------------------------
def get_last_assistant_content(resp):
    if isinstance(resp, tuple): resp = resp[0]
    if not isinstance(resp, list): return ""
    for turn in reversed(resp):
        if turn.get("role")!='assistant': continue
        if turn.get("content"): return turn["content"]
        fr=turn.get("function_response",{})
        out=fr.get("result",{}).get("output")
        if out: return out
        cont=turn.get("content")
        if isinstance(cont,dict): parts=cont.get("parts",[])
        if parts and parts[0].get("text"): return parts[0]["text"]
    return ""

# -----------------------------------------------------------------------------
# Play one automated Globle game via LLM
# -----------------------------------------------------------------------------
def play_globle_agent(client, countries, geoms, max_guesses=20, threshold=0.6):
    # pick random target
    target, (tlat, tlon) = random.choice(list(countries.items()))
    guesses = []
    attempts = 0

    while True:
        # build prompt history
        history = "\n".join([f"Guess: {g}, Response: {resp}" for g,resp in guesses])
        prompt = (
            "Worldle (distance-only). Guess the country.\n" +
            (history+"\n" if history else "") +
            "Respond with a single country name and ONLY the name of the country."
        )
        resp = client.predict(messages=[{"role":"user","content":prompt}], api_name="/run")
        raw = get_last_assistant_content(resp).strip()
        print(f"Guess: {raw}")
        # sanitize: fuzzy match to known country
        if raw not in countries:
            match = difflib.get_close_matches(raw, countries.keys(), n=1, cutoff=threshold)
            if match:
                guess = match[0]
            else:
                # invalid guess, retry without counting
                continue
        else:
            guess = raw

        attempts += 1
        # correct?
        if guess == target:
            return {"target":target, "guesses":guesses+[(guess,"CORRECT")], "turns":attempts, "solved":True}

        # adjacency
        if geoms[guess].touches(geoms[target]):
            feedback="BORDER"
        else:
            # distance
            glat, glon = countries[guess]
            dist = haversine(glat, glon, tlat, tlon)
            feedback=f"{dist:.0f}km"
        guesses.append((guess,feedback))
        if attempts>=max_guesses:
            return {"target":target, "guesses":guesses, "turns":attempts, "solved":False}

# -----------------------------------------------------------------------------
# Benchmark multiple games
# -----------------------------------------------------------------------------
def benchmark_globle(geo_path, num_games=1, max_guesses=20, cutoff=0.6):
    countries, geoms = load_countries(geo_path)
    client = Client("http://127.0.0.1:7860/")
    os.makedirs("results",exist_ok=True)
    out_file = os.path.join("results", f"globle_benchmark_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    results=[]
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        start=time.time()
        res=play_globle_agent(client,countries,geoms,max_guesses,cutoff)
        res["time"] = time.time()-start
        results.append(res)
        with open(out_file,"a") as f: f.write(json.dumps(res)+"\n")
    print(f"Saved results to {out_file}")
    return results

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python benchmarking_globle.py path/to/countries-file")
        sys.exit(1)
    geo=sys.argv[1]
    benchmark_globle(geo)
