import importlib

__all__ = ['GoogleSearchTool']


class GoogleSearchTool():
    dependencies = ["googlesearch-python==1.3.0"]

    inputSchema = {
        "name": "GoogleSearchTool",
        "description": "Provides a list of URLs from google search results based on a query string. Use the urls then to get the content of the page.",
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
            "required": ["query"],
        }
    }

    def run(self, **kwargs):
        print("Running web search")

        website = kwargs.get("website")
        query = kwargs.get("query")

        if not query:
            return {
                "status": "error",
                "message": "Missing required parameters: 'query'",
                "output": None
            }
        search_query = query
        if website:
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
