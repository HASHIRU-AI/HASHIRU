import importlib
import random
import math
import difflib
import json
import os
from pathlib import Path

__all__ = ["GlobleDistanceTool"]


class GlobleDistanceTool:
    """Globle‑style geography guessing game.

    *Persists* game state to a JSON file in the user’s home directory (`~/.Globle_distance_state.json`).
    This survives process restarts, ensuring the hidden country stays the same
    until the player guesses correctly, gives up, or passes `reset=True`.
    """

    # ------------------------------------------------------------------
    # Metadata for orchestration
    # ------------------------------------------------------------------
    dependencies = [
        "geopandas==1.0.1",
        "shapely==2.1.0",
    ]

    inputSchema = {
        "name": "GlobleDistanceTool",
        "description": (
            "Guess the hidden country. Tool replies with distance to the target's centroid, "
            "or notifies if they share a border. Use '/GIVEUP' to surrender."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "geo_path": {
                    "type": "string",
                    "description": "Path to GeoJSON/Shapefile with country polygons. Default './tools/util/countries.geojson'.",
                },
                "guess": {
                    "type": "string",
                    "description": "The country you are guessing (case‑insensitive).",
                },
                "reset": {
                    "type": "boolean",
                    "description": "Force a new game and discard stored state.",
                },
            },
        },
    }

    # ------------------------------------------------------------------
    # Constants / state file location
    # ------------------------------------------------------------------
    _STATE_PATH = Path.home() / ".Globle_distance_state.json"

    # Runtime caches (populated lazily)
    _countries_cache = {}  # geo_path → (countries, geoms)

    # --------------------------- public entry ---------------------------
    def run(self, **kwargs):
        geo_path = kwargs.get("geo_path", "./tools/util/countries.geojson")
        guess = kwargs.get("guess")
        reset_flag = bool(kwargs.get("reset", False))

        # -----------------------------------
        # Load or reset persistent state file
        # -----------------------------------
        if reset_flag and self._STATE_PATH.exists():
            self._STATE_PATH.unlink(missing_ok=True)

        state = self._load_state()
        if state is None:  # no current game → start one
            init_res = self._start_new_game(geo_path)
            if init_res["status"] != "success":
                return init_res
            state = self._load_state()  # reload the freshly written state

        # ------------------
        # Ensure guess given
        # ------------------
        if not guess:
            return {
                "status": "error",
                "message": "Provide a 'guess' parameter, or '/GIVEUP' to give up.",
                "output": None,
            }

        # --------------------------------------------------------------
        # Play round
        # --------------------------------------------------------------
        countries, geoms = self._get_country_data(state["geo_path"])
        target = state["target"]
        tlat, tlon = countries[target]

        raw_guess = guess.strip()

        # Handle give‑up
        if raw_guess.upper() == "/GIVEUP":
            self._STATE_PATH.unlink(missing_ok=True)
            return {
                "status": "success",
                "message": "Gave up.",
                "output": {"result": "gave_up", "answer": target},
            }

        # Fuzzy match unknown country
        if raw_guess not in countries:
            match = difflib.get_close_matches(raw_guess, countries.keys(), n=1, cutoff=0.6)
            if not match:
                return {
                    "status": "error",
                    "message": f"Unknown country '{raw_guess}'.",
                    "output": None,
                }
            guess_name = match[0]
        else:
            guess_name = raw_guess

        # Correct?
        if guess_name == target:
            self._STATE_PATH.unlink(missing_ok=True)  # clear state for next game
            return {
                "status": "success",
                "message": "Correct!",
                "output": {"result": "correct", "country": target},
            }

        # Shared border?
        if geoms[guess_name].touches(geoms[target]):
            return {
                "status": "success",
                "message": "Bordering country.",
                "output": {"result": "border", "country": guess_name},
            }

        # Distance feedback
        glat, glon = countries[guess_name]
        dist_km = self._haversine(glat, glon, tlat, tlon)
        return {
            "status": "success",
            "message": "Distance computed.",
            "output": {"result": "distance", "country": guess_name, "km": round(dist_km)},
        }

    # ------------------------------------------------------------------
    # Helper: start new game and persist state
    # ------------------------------------------------------------------
    def _start_new_game(self, geo_path):
        try:
            countries, _ = self._get_country_data(geo_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load '{geo_path}': {e}",
                "output": None,
            }
        target = random.choice(list(countries))
        state = {"geo_path": geo_path, "target": target}
        try:
            self._STATE_PATH.write_text(json.dumps(state))
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot write state file: {e}",
                "output": None,
            }
        return {"status": "success", "message": "New game started", "output": None}

    # ------------------------------------------------------------------
    # Helper: load state file (or None if missing)
    # ------------------------------------------------------------------
    def _load_state(self):
        if not self._STATE_PATH.exists():
            return None
        try:
            return json.loads(self._STATE_PATH.read_text())
        except Exception:
            # Corrupt state – delete and start fresh next time
            self._STATE_PATH.unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Country data cache (per geo_path) to avoid re‑reading file
    # ------------------------------------------------------------------
    def _get_country_data(self, geo_path):
        if geo_path in self._countries_cache:
            return self._countries_cache[geo_path]
        countries, geoms = self._load_countries(geo_path)
        self._countries_cache[geo_path] = (countries, geoms)
        return countries, geoms

    # ------------------------------------------------------------------
    # Geometry / file utility
    # ------------------------------------------------------------------
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def _load_countries(geo_path):
        gpd = importlib.import_module("geopandas")
        Point = importlib.import_module("shapely.geometry").Point
        gdf = gpd.read_file(geo_path)

        name_field = next((c for c in ["ADMIN", "NAME", "NAME_EN", "NAME_LONG", "SOVEREIGN", "COUNTRY"] if c in gdf.columns), None)
        if not name_field:
            non_geom = [c for c in gdf.columns if c.lower() != "geometry"]
            if not non_geom:
                raise ValueError("No suitable name column in geo file")
            name_field = non_geom[0]

        centroids, geoms = {}, {}
        for _, row in gdf.iterrows():
            geom = row.geometry
            if not geom or geom.is_empty:
                continue
            c = geom.centroid  # type: Point
            name = row[name_field]
            centroids[name] = (c.y, c.x)
            geoms[name] = geom
        return centroids, geoms
