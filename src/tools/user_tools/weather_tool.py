import importlib
import os

__all__ = ['WeatherApi']


class WeatherApi():
    dependencies = ["requests==2.32.3"]

    inputSchema = {
        "name": "WeatherApi",
        "description": "Returns weather information for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location for which to get the weather information",
                },
            },
            "required": ["location"],
        },
        "invoke_resource_cost": 0.1,
    }

    def run(self, **kwargs):
        print("Running Weather API test tool")
        location = kwargs.get("location")
        print(f"Location: {location}")

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "Weather API test tool failed",
                "error": "OPENWEATHER_API_KEY environment variable is not set",
                "output": None
            }

        requests = importlib.import_module("requests")

        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": api_key})
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Weather API test tool executed successfully",
                "error": None,
                "output": response.json()
            }
        else:
            return {
                "status": "error",
                "message": "Weather API test tool failed",
                "error": response.text,
                "output": None
            }
