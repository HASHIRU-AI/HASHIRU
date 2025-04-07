import importlib

__all__ = ['WebSearchTool']


class WebSearchTool():
    dependencies = ["googlesearch-python==1.3.0"]

    inputSchema = {
        "name": "WebSearchTool",
        "description": "Searches a specific website for a given query using Google search.",
        "parameters": {
            "type": "object",
            "properties": {
                "website": {
                    "type": "string",
                    "description": "The website domain to search in (e.g., 'stackoverflow.com').",
                },
                "query": {
                    "type": "string",
                    "description": "The query string to search for on the website.",
                }
            },
            "required": ["website", "query"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        print("Running web search")

        website = kwargs.get("website")
        query = kwargs.get("query")

        if not website or not query:
            return {
                "status": "error",
                "message": "Missing required parameters: 'website' and 'query'",
                "output": None
            }

        search_query = f"site:{website} {query}"
        results = []

        googlesearch = importlib.import_module("googlesearch")

        try:
            for result in googlesearch.search(search_query, num_results=10):
                if "/search?num=" not in result:
                    results.append(result)

            return {
                "status": "success",
                "message": "Search completed successfully",
                "output": results,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "output": None,
            }
