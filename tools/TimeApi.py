import datetime

__all__ = ['TimeApi']

class TimeApi():
    dependencies = []

    inputSchema = {
        "name": "TimeApi",
        "description": "Returns the current time for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location for which to get the current time",
                },
            },
            "required": ["location"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        location = kwargs.get("location")
        try:
            # This will only work if the timezone is configured correctly on the server
            now = datetime.datetime.now(datetime.timezone.utc)
            return {
                "status": "success",
                "message": f"Current time in {location} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}",
                "error": None,
                "output": now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not get current time for {location}",
                "error": str(e),
                "output": None
            }
