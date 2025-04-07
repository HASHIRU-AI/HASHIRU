import importlib

__all__ = ['GetWebsiteTool']


class GetWebsiteTool():
    dependencies = ["requests"]

    inputSchema = {
        "name": "GetWebsiteTool",
        "description": "Returns the content of a website based on a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to fetch content from.",
                },
            },
            "required": ["url"],
        }
    }

    def run(self, **kwargs):
        print("Running web search")

        url = kwargs.get("url")

        if not url:
            return {
                "status": "error",
                "message": "Missing required parameters: 'url'",
                "output": None
            }
        
        output = None
        requests = importlib.import_module("requests")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                output = response.text
            else:
                return {
                    "status": "error",
                    "message": f"Failed to fetch content from {url}. Status code: {response.status_code}",
                    "output": None
                }
            
            # truncate the results to avoid excessive output
            if len(output) > 1000:
                output = output[:1000] + "... (truncated)"

            return {
                "status": "success",
                "message": "Search completed successfully",
                "output": output,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None
            }
