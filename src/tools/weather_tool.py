import importlib

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
        "invoke_cost": 0.1,
    }

    def run(self, **kwargs):
        print("Running Weather API test tool")
        location = kwargs.get("location")
        print(f"Location: {location}")

        requests = importlib.import_module("requests")

        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid=ea50e63a3bea67adaf50fbecbe5b3c1e")
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
